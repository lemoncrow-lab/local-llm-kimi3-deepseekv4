"""Run DeepSeek-V4-Flash-0731 on a single RTX 4090 by streaming the routed experts.

The checkpoint is 155.42 GiB.  The always-active trunk -- attention, shared experts,
router, head -- is only 7.83 GiB and lives on the GPU; the 146.62 GiB routed-expert
pool stays on disk, mmap'd, and only the 43 x 6 experts a token actually selects are
fetched (3.21 GiB/token, two contiguous reads each).

DeepSeek's own inference/model.py supplies all the math.  Two things are swapped out:
its tilelang kernels (which need shared memory Ada does not have) for the torch
stand-ins in kernels_torch.py, and MoE.forward for a streaming version.
"""
import argparse
import json
import mmap
import os
import struct
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("DSV4_MODEL", "/home/pankaj/Models/DeepSeek-V4-Flash-0731")
REF = os.path.join(MODEL_DIR, "inference")
sys.path.insert(0, HERE)
sys.path.insert(0, REF)
sys.path.insert(0, os.path.join(MODEL_DIR, "encoding"))

import kernels_torch                                    # noqa: E402
from stream import ExpertStream                         # noqa: E402

sys.modules["kernel"] = kernels_torch                   # model.py does `from kernel import ...`
sys.modules["fast_hadamard_transform"] = kernels_torch  # ... and imports hadamard_transform

# The torch stand-ins are the reference; these triton versions are what actually runs.
# DSV4_TRITON=0 falls back to pure torch (~2.5x slower per token, same numbers).
if os.environ.get("DSV4_TRITON", "1") != "0":
    import fp4_triton                                   # noqa: E402
    import tk                                           # noqa: E402

    for _n in ("act_quant", "fp4_act_quant", "fp8_gemm", "hc_split_sinkhorn"):
        setattr(kernels_torch, _n, getattr(tk, _n))
    kernels_torch.fp4_gemm = fp4_triton.fp4_gemm

# Bisect aid: DSV4_TILELANG=act_quant,fp4_gemm,... swaps named functions back to the
# checkpoint's own tilelang kernels (needs CUDA_HOME=/home/pankaj/Models/.cuda13).
# sparse_attn is not swappable -- it does not fit in Ada's shared memory.
if os.environ.get("DSV4_TILELANG"):
    import importlib

    _tl = importlib.import_module("kernel_tilelang")
    for _name in os.environ["DSV4_TILELANG"].split(","):
        _name = _name.strip()
        if _name:
            setattr(kernels_torch, _name, getattr(_tl, _name))
            print(f"tilelang: {_name}", flush=True)


class Weights:
    """mmap'd view of the safetensors shards, by tensor name."""

    DTYPES = {
        "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
        "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.uint8, "I32": torch.int32, "I64": torch.int64,
    }

    def __init__(self, model_dir):
        self.index = {}
        self.maps = {}
        for fn in sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors")):
            path = os.path.join(model_dir, fn)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.index[k] = (path, base + a, b - a, v["dtype"], tuple(v["shape"]))

    def _map(self, path):
        if path not in self.maps:
            fh = open(path, "rb")
            self.maps[path] = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
            fh.close()
        return self.maps[path]

    def get(self, name, device="cpu"):
        path, off, nbytes, dt, shape = self.index[name]
        raw = torch.frombuffer(self._map(path), dtype=torch.uint8, count=nbytes, offset=off)
        if device != "cpu":
            raw = raw.to(device, non_blocking=True)
        t = raw.view(self.DTYPES[dt])
        # fp4 is stored packed: [out, in] logical, [out, in//2] bytes
        return t.view(*shape[:-1], -1) if dt == "I8" else t.view(shape)

    def __contains__(self, name):
        return name in self.index


def load_trunk(model, w, device="cuda"):
    """Copy every non-expert tensor into the constructed model."""
    have = dict(model.named_parameters())
    have.update(dict(model.named_buffers()))
    loaded, missing = 0, []
    for name, p in have.items():
        if ".experts." in name and "shared_experts" not in name:
            continue
        if name not in w:
            if not name.endswith(("kv_cache", "freqs_cis", "kv_state", "score_state")):
                missing.append(name)      # the rest are computed buffers, not checkpoint data
            continue
        src = w.get(name, device)
        if name.endswith("wo_a.weight"):
            # convert.py dequantises this one to bf16 at conversion time, and
            # Attention.forward then uses the raw weight in an einsum with no scale --
            # so the block scale has to be folded in here or attention comes out wrong.
            sc = w.get(name.replace("weight", "scale"), device).float()
            src = (src.float().unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
                   * sc[:, None, :, None]).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)
        elif src.dtype != p.dtype and p.dtype in (torch.float32, torch.bfloat16):
            src = src.to(p.dtype)
        p.data = src.view(p.shape) if src.numel() == p.numel() else src
        loaded += 1
    return loaded, missing


def install_streaming_moe(model, stream, args):
    """Replace MoE.forward with the streaming, grouped-fp4 version.

    Decode (one token) routes to `topk` experts that all see the same activation row,
    so the whole layer is two kernel launches: [E,1,K] x [E,2*inter,K] for w1|w3, then
    [E,1,inter] x [E,dim,inter] for w2.  Prefill keeps a per-expert loop -- token counts
    differ per expert -- but still fetches the layer's experts in one batched pull.
    """
    import torch.nn.functional as F

    import model as M
    from fp4_triton import fp4_gemm_grouped
    from kernels_torch import act_quant

    limit = args.swiglu_limit
    inter = args.moe_inter_dim
    e8m0 = torch.float8_e8m0fnu

    def swiglu(h):
        g, u = h[..., :inter].float(), h[..., inter:].float()
        if limit > 0:
            u = u.clamp(-limit, limit)
            g = g.clamp(max=limit)
        return F.silu(g) * u

    def moe_forward(self, x, input_ids):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)

        if x.size(0) == 1:                                  # decode
            experts = indices.view(-1).tolist()
            idx, (W13, S13, W2, S2) = stream.get_layer(self.layer_id, experts)
            t0 = time.perf_counter()
            xq, xs = act_quant(x, 128, "ue8m0", e8m0)
            h = swiglu(fp4_gemm_grouped(xq[None], xs[None], W13, S13, idx=idx))
            hq, hs = act_quant(h.to(x.dtype), 128, "ue8m0", e8m0)
            o = fp4_gemm_grouped(hq, hs, W2, S2, idx=idx)   # [E, 1, dim]
            y += (o.float() * weights.view(-1, 1, 1).float()).sum(0)
            stream.release()
            stream.t_expert += time.perf_counter() - t0
        else:                                               # prefill
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
            active = torch.nonzero(counts).flatten().tolist()
            for c0 in range(0, len(active), stream.E):
                chunk = active[c0:c0 + stream.E]
                sidx, (W13, S13, W2, S2) = stream.get_layer(self.layer_id, chunk)
                t0 = time.perf_counter()
                for j, e in enumerate(chunk):
                    rows, top = torch.where(indices == e)
                    xq, xs = act_quant(x[rows], 128, "ue8m0", e8m0)
                    h = swiglu(fp4_gemm_grouped(xq[None], xs[None], W13, S13,
                                                idx=sidx[j:j + 1])[0])
                    hq, hs = act_quant(h.to(x.dtype), 128, "ue8m0", e8m0)
                    o = fp4_gemm_grouped(hq[None], hs[None], W2, S2, idx=sidx[j:j + 1])[0]
                    y[rows] += (weights[rows, top, None] * o).float()
                stream.release()
                stream.t_expert += time.perf_counter() - t0

        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    M.MoE.forward = moe_forward

def build(args_cfg, seq_len, device="cuda"):
    """Construct the model with the routed experts left unallocated."""
    import model as M

    orig_init = M.MoE.__init__

    def moe_init(self, layer_id, cfg):
        n = cfg.n_routed_experts
        cfg.n_routed_experts = 0                        # skip 256 x Expert allocation
        orig_init(self, layer_id, cfg)
        cfg.n_routed_experts = n
        self.n_routed_experts = self.n_local_experts = n
        self.experts_end_idx = n
        self.experts = torch.nn.ModuleList()

    M.MoE.__init__ = moe_init
    try:
        with torch.device(device):
            model = M.Transformer(args_cfg)
    finally:
        M.MoE.__init__ = orig_init
    return model


def try_server(a):
    """If serve-dsv4-4090.sh is already up, send the prompt there.

    The GPU fits exactly one copy of the trunk, so a CLI run alongside a server used to
    die with an OOM 40 frames deep.  It is also simply the better path: the server's
    expert caches are already warm, so there is no ~35 s cold start.

    DSV4_NO_SERVER=1 forces a local load; DSV4_SERVER / DSV4_PORT point elsewhere.
    """
    import urllib.request

    if os.environ.get("DSV4_NO_SERVER") or a.ids:
        return False
    base = os.environ.get("DSV4_SERVER") or \
        f"http://127.0.0.1:{os.environ.get('DSV4_PORT', 8000)}"
    try:
        # not just "something answers": another service may own the port
        if b"deepseek-v4-flash" not in urllib.request.urlopen(base + "/v1/models",
                                                              timeout=1.5).read():
            return False
    except Exception:
        return False

    if a.chat:
        path = "/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": a.prompt}],
                "thinking_mode": a.thinking, "reasoning_effort": a.effort}
    else:
        path = "/v1/completions"
        body = {"prompt": a.prompt}
    body.update(max_tokens=a.max_new_tokens, temperature=a.temperature, stream=True)
    print(f"server: {base} (DSV4_NO_SERVER=1 to load locally instead)",
          file=sys.stderr, flush=True)

    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft, n = None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            ch = json.loads(line[6:])["choices"][0]
            piece = ch.get("delta", {}).get("content") or ch.get("text") or ""
            if piece:
                n += 1                      # one SSE delta == one decoded token
                if ttft is None:
                    ttft = time.perf_counter() - t0
            sys.stdout.write(piece)
            sys.stdout.flush()
    print()
    dt = time.perf_counter() - t0
    if n:
        # decode rate excludes the prefill: TTFT covers the prompt.
        dec = (n - 1) / (dt - ttft) if n > 1 and dt > ttft else float("nan")
        print(f"{n} tokens in {dt:.1f}s -- {dec:.2f} tok/s decode, "
              f"{ttft:.1f}s to first token", file=sys.stderr, flush=True)
    return True


def preflight_vram(need_gib=13.0):
    """Fail with the culprit named instead of a torch OOM 40 frames deep.

    The trunk alone is 11.9 GiB, so a resident serve-dsv4-4090.sh (which takes every
    spare byte for its expert cache) leaves nothing for a second process.
    """
    import subprocess
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free >= need_gib:
        return
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        apps = ""
    mine = os.getpid()
    others = [ln for ln in apps.splitlines() if ln and int(ln.split(",")[0]) != mine]
    detail = ""
    for ln in others:
        pid = ln.split(",")[0].strip()
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        detail += f"\n  pid {pid} holds {ln.split(',')[1].strip()}: {cmd[:100]}"
    sys.exit(f"only {free:.1f} GiB of VRAM free, need ~{need_gib:.0f} GiB.{detail}\n"
             f"The CLI and the server cannot share the GPU -- stop one of them.\n"
             f"If that pid is serve-dsv4-4090.sh, point the CLI at it instead:\n"
             f"  DSV4_PORT=<its port> {os.path.basename(sys.argv[0])} ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="The sky is blue because")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ids", type=int, nargs="*", help="raw token ids, skips the tokenizer")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat encoding")
    ap.add_argument("--thinking", default="chat", help="thinking_mode for the chat encoding")
    ap.add_argument("--effort", default="low", help="reasoning_effort for the chat encoding")
    a = ap.parse_args()

    if try_server(a):
        return

    torch.set_default_dtype(torch.bfloat16)
    torch.set_grad_enabled(False)
    preflight_vram()

    import model as M
    with open(os.path.join(REF, "config.json")) as f:
        cfg = M.ModelArgs(**json.load(f))
    cfg.max_batch_size = 1
    cfg.max_seq_len = a.seq_len
    cfg.temperature = a.temperature

    t0 = time.perf_counter()
    w = Weights(MODEL_DIR)
    print(f"index: {len(w.index)} tensors in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    model = build(cfg, a.seq_len)
    print(f"construct: {time.perf_counter()-t0:.1f}s, "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GiB on gpu", flush=True)

    t0 = time.perf_counter()
    loaded, missing = load_trunk(model, w)
    torch.cuda.synchronize()
    print(f"trunk: {loaded} tensors in {time.perf_counter()-t0:.1f}s, "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GiB on gpu", flush=True)
    if missing:
        print(f"  MISSING {len(missing)}: {missing[:8]}", flush=True)

    torch.cuda.empty_cache()      # hand the loader's temporaries back before sizing
    stream = ExpertStream(w, vram_gib=float(os.environ.get("DSV4_VRAM_CACHE_GIB", 0)),
                          ram_gib=float(os.environ.get("DSV4_RAM_CACHE_GIB", 24.0)),
                          threads=int(os.environ.get("DSV4_IO_THREADS", 32)),
                          max_experts=int(os.environ.get("DSV4_FETCH_EXPERTS", 64)))
    print(f"cache: {stream.Ng} experts in vram ({stream.Ng*12.75/1024:.1f} GiB), "
          f"{stream.Nh} pinned in ram ({stream.Nh*12.75/1024:.1f} GiB)", flush=True)
    install_streaming_moe(model, stream, cfg)
    torch.set_default_device("cuda")          # index helpers in model.py allocate bare tensors

    if a.ids:
        ids = a.ids
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        text = a.prompt
        if a.chat:
            # the released model is post-trained; raw text is out of distribution
            from encoding_dsv4 import encode_messages
            text = encode_messages([{"role": "user", "content": a.prompt}],
                                   thinking_mode=a.thinking, reasoning_effort=a.effort)
        ids = tok.encode(text)
    print(f"prompt: {len(ids)} tokens {ids[:16]}", flush=True)

    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    out = []
    prev = 0
    for step in range(a.max_new_tokens):
        t0 = time.perf_counter()
        b0 = stream.bytes
        c0 = (stream.hit_g, stream.hit_h, stream.miss)
        tf0, te0 = stream.t_fetch, stream.t_expert
        nxt = model.forward(tokens[:, prev:], prev)[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        prev = tokens.size(1)
        tokens = torch.cat([tokens, nxt.view(1, 1)], dim=1)
        out.append(int(nxt.item()))
        if out[-1] == 1:                      # eos_token_id
            print("eos", flush=True)
            break
        print(f"step {step}: {dt:7.2f}s  {1/dt:5.3f} tok/s  "
              f"{(stream.bytes-b0)/2**30:5.2f} GiB experts  "
              f"cache {stream.hit_g-c0[0]:3d}v/{stream.hit_h-c0[1]:3d}r/{stream.miss-c0[2]:3d}d  "
              f"[fetch {stream.t_fetch-tf0:5.2f}s  expert-gemm {stream.t_expert-te0:5.2f}s  "
              f"rest {dt-(stream.t_fetch-tf0)-(stream.t_expert-te0):5.2f}s]  "
              f"id={out[-1]}", flush=True)

    print("ids:", out)
    if not a.ids:
        print("text:", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
