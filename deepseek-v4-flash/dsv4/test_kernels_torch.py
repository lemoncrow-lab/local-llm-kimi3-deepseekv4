"""Check the torch stand-ins against the semantics in inference/kernel.py.

Run:  /home/pankaj/Models/.venv-dsv4/bin/python dsv4/test_kernels_torch.py
"""
import sys

import torch

sys.path.insert(0, "/home/pankaj/Models/dsv4")
import kernels_torch as K

torch.manual_seed(20260803)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

fails = []


def ok(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# --- quantisation round trips -------------------------------------------------
x = torch.randn(64, 4096)
xq, xs = K.act_quant(x, 128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
xdq = K.act_quant(x.clone(), 128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu, inplace=True)
ok("act_quant round trip", rel(xdq, x) < 0.05, f"rel={rel(xdq, x):.4f}")
ok("act_quant scales pow2", bool((torch.log2(xs.float()) % 1 == 0).all()))
ok("act_quant shapes", xq.shape == x.shape and tuple(xs.shape) == (64, 32) and xq.dtype == torch.float8_e4m3fn)

w = torch.randn(256, 4096)
wq, ws = K.fp4_act_quant(w, 32)
wdq = K.fp4_act_quant(w.clone(), 32, inplace=True)
ok("fp4_act_quant round trip", rel(wdq, w) < 0.20, f"rel={rel(wdq, w):.4f}")
ok("fp4_act_quant scales pow2", bool((torch.log2(ws.float()) % 1 == 0).all()))
ok("fp4_act_quant packing", tuple(wq.shape) == (256, 2048) and wq.dtype == torch.float4_e2m1fn_x2)
ok("fp4 unpack == fake quant", rel(K.fp4_unpack(wq) * ws.float().repeat_interleave(32, -1), wdq) < 1e-6)

# --- GEMMs --------------------------------------------------------------------
for M in (1, 32, 128):
    a = torch.randn(M, 4096)
    aq, as_ = K.act_quant(a, 128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    adq = K.act_quant(a.clone(), 128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu, inplace=True)
    y = K.fp4_gemm(aq, as_, wq, ws, scale_dtype=torch.float8_e8m0fnu)
    ref = adq.float() @ wdq.float().T
    ok(f"fp4_gemm M={M}", rel(y, ref) < 0.02, f"rel={rel(y, ref):.5f}")

# fp8 weights use one scale per 128x128 block
wf = torch.randn(2048, 4096)
g = wf.float().unflatten(1, (32, 128)).unflatten(0, (16, 128))       # [16,128,32,128]
amax = g.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
sb = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
wf8 = (g / sb).clamp(-448, 448).to(torch.float8_e4m3fn)
wf_dq = (wf8.float() * sb).permute(0, 1, 2, 3).reshape(2048, 4096)
wf8 = wf8.reshape(2048, 4096).contiguous()
sb = sb.reshape(16, 32).to(torch.float8_e8m0fnu).contiguous()
for M in (1, 32):
    a = torch.randn(M, 4096)
    aq, as_ = K.act_quant(a, 128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    adq = K.act_quant(a.clone(), 128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu, inplace=True)
    y = K.fp8_gemm(aq, as_, wf8, sb, scale_dtype=torch.float8_e8m0fnu)
    ref = adq.float() @ wf_dq.float().T
    ok(f"fp8_gemm M={M}", rel(y, ref) < 0.02, f"rel={rel(y, ref):.5f}")

# --- sparse attention ---------------------------------------------------------
b, s, h, d, L, topk = 2, 3, 4, 64, 128, 16
q = torch.randn(b, s, h, d)
kv = torch.randn(b, L, d)
sink = torch.randn(h, dtype=torch.float32)
idx = torch.randint(0, L, (b, s, topk), dtype=torch.int32)
idx[0, 0, :4] = -1
scale = d ** -0.5
o = K.sparse_attn(q, kv, sink, idx, scale)

gather = kv[torch.arange(b).view(b, 1, 1), idx.clamp_min(0).long()].float()
sc = torch.einsum("bshd,bskd->bshk", q.float(), gather) * scale
sc = sc.masked_fill(~(idx >= 0).unsqueeze(2), float("-inf"))
mx = sc.amax(-1, keepdim=True)
p = torch.exp(sc - mx)
den = p.sum(-1, keepdim=True) + torch.exp(sink.view(1, 1, h, 1) - mx)
ref = torch.einsum("bshk,bskd->bshd", p, gather) / den
ok("sparse_attn vs einsum", rel(o, ref) < 1e-2, f"rel={rel(o, ref):.5f}")
ok("sparse_attn masks -1", bool(torch.isfinite(o).all()))

q = torch.randn(1, 1, 64, 512)
kv = torch.randn(1, 4096, 512)
idx = torch.randint(0, 4096, (1, 1, 512), dtype=torch.int32)
o = K.sparse_attn(q, kv, torch.randn(64, dtype=torch.float32), idx, 512 ** -0.5)
ok("sparse_attn decode shape", tuple(o.shape) == (1, 1, 64, 512) and bool(torch.isfinite(o).all()))

# --- hyper-connection sinkhorn ------------------------------------------------
hc = 4
mixes = torch.randn(2, 5, (2 + hc) * hc, dtype=torch.float32)
pre, post, comb = K.hc_split_sinkhorn(mixes, torch.randn(3, dtype=torch.float32),
                                      torch.randn((2 + hc) * hc, dtype=torch.float32), hc, 20, 1e-6)
ok("sinkhorn shapes", tuple(pre.shape) == (2, 5, hc) and tuple(comb.shape) == (2, 5, hc, hc))
ok("sinkhorn doubly stochastic", float((comb.sum(-2) - 1).abs().max()) < 1e-3,
   f"col max dev={float((comb.sum(-2) - 1).abs().max()):.2e}")
ok("sinkhorn pre>0", bool((pre > 0).all()) and bool((post >= 0).all()))

print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
