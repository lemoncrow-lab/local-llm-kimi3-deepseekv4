#!/usr/bin/env python
"""tk.py (triton) must match kernels_torch on relative error, not correlation."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels_torch as KT      # noqa: E402
import tk as TK                 # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
E8 = torch.float8_e8m0fnu
fail = []


def check(name, got, want, tol=5e-3):
    g, w = got.float(), want.float()
    r = ((g - w).norm() / w.norm().clamp_min(1e-9)).item()
    ok = r < tol
    print(f"{'ok ' if ok else 'FAIL'} {name:34s} rel={r:.6f}")
    if not ok:
        fail.append(name)


# --- hc_split_sinkhorn
for b, s in ((1, 1), (1, 9), (2, 40)):
    mix = torch.randn(b, s, 24, device=dev, dtype=torch.bfloat16)
    sc = torch.randn(3, device=dev, dtype=torch.float32).abs()
    ba = torch.randn(24, device=dev, dtype=torch.float32)
    for i, nm in enumerate(("pre", "post", "comb")):
        check(f"sinkhorn b={b} s={s} {nm}",
              TK.hc_split_sinkhorn(mix, sc, ba)[i], KT.hc_split_sinkhorn(mix, sc, ba)[i])

# --- act_quant, out of place
for shape, bs in (((1, 4096), 128), ((9, 4096), 128), ((6, 1, 2048), 128)):
    x = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
    gq, gs = TK.act_quant(x, bs, "ue8m0", E8)
    wq, ws = KT.act_quant(x, bs, "ue8m0", E8)
    check(f"act_quant{shape} q", gq.float(), wq.float())
    check(f"act_quant{shape} s", gs.float(), ws.float())

# --- act_quant, in place, on a non-contiguous last-dim slice (model.py does this)
kv = torch.randn(1, 7, 576, device=dev, dtype=torch.bfloat16)
a, b_ = kv.clone(), kv.clone()
TK.act_quant(a[..., :-64], 64, "ue8m0", E8, True)
KT.act_quant(b_[..., :-64], 64, "ue8m0", E8, True)
check("act_quant inplace slice", a, b_)

# --- fp4_act_quant, in place
a, b_ = kv.clone(), kv.clone()
TK.fp4_act_quant(a[..., :-64], 32, True)
KT.fp4_act_quant(b_[..., :-64], 32, True)
check("fp4_act_quant inplace slice", a, b_)


# --- fp8_gemm
def rand_fp8(N, K):
    w = torch.randn(N, K, device=dev) * 0.05
    q, s = KT.act_quant(w.view(N // 128, 128, K // 128, 128).permute(0, 2, 1, 3).reshape(-1, 128 * 128),
                        128 * 128, "ue8m0", E8)
    q = q.view(N // 128, K // 128, 128, 128).permute(0, 2, 1, 3).reshape(N, K)
    return q.contiguous(), s.view(N // 128, K // 128).contiguous()


for M in (1, 9, 64, 257):
    for N, K in ((4096, 4096), (32768, 1024), (512, 4096)):
        b, bs = rand_fp8(N, K)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        aq, asq = KT.act_quant(x, 128, "ue8m0", E8)
        check(f"fp8_gemm M={M} N={N} K={K}",
              TK.fp8_gemm(aq, asq, b, bs, E8), KT.fp8_gemm(aq, asq, b, bs, E8))

print("FAILED: " + ", ".join(fail) if fail else "ALL PASS")
sys.exit(1 if fail else 0)
