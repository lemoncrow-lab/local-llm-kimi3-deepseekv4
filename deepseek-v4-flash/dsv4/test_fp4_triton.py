#!/usr/bin/env python
"""fp4_triton must match the torch reference bit-for-bit-ish, not just correlate."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels_torch as KT      # noqa: E402
import fp4_triton as FT         # noqa: E402

torch.manual_seed(0)
dev = "cuda"
E8 = torch.float8_e8m0fnu
fail = []


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()


def rand_fp4(N, K):
    """Random weights already on the fp4 grid, with real e8m0 block scales."""
    w = torch.randn(N, K, device=dev) * 0.05
    packed, scale = KT.fp4_act_quant(w, 32)
    return packed.view(torch.uint8), scale


def check(name, got, want, tol=5e-3):
    r = rel(got, want)
    ok = r < tol
    print(f"{'ok ' if ok else 'FAIL'} {name:28s} rel={r:.5f}")
    if not ok:
        fail.append(name)


for M in (1, 4, 32, 129):
    for N, K in ((4096, 4096), (4096, 2048), (2048, 4096)):
        b, bs = rand_fp4(N, K)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        aq, asq = KT.act_quant(x, 128, "ue8m0", E8)
        want = KT.fp4_gemm(aq, asq, b.view(torch.float4_e2m1fn_x2), bs, E8)
        got = FT.fp4_gemm(aq, asq, b, bs, E8)
        check(f"M={M} N={N} K={K}", got, want)

# grouped, with A broadcast across the group -- the decode path
Eg, M, N, K = 6, 1, 4096, 4096
bs_l, b_l = [], []
for _ in range(Eg):
    p, s = rand_fp4(N, K)
    b_l.append(p)
    bs_l.append(s)
B = torch.stack(b_l)
BS = torch.stack(bs_l)
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
aq, asq = KT.act_quant(x, 128, "ue8m0", E8)
got = FT.fp4_gemm_grouped(aq[None], asq[None], B, BS)
want = torch.stack([KT.fp4_gemm(aq, asq, b_l[e].view(torch.float4_e2m1fn_x2), bs_l[e], E8)
                    for e in range(Eg)])
check("grouped E=6 broadcast A", got, want)

# grouped with a distinct A per expert -- the w2 path
A = torch.randn(Eg, M, K, device=dev, dtype=torch.bfloat16)
aq, asq = KT.act_quant(A, 128, "ue8m0", E8)
got = FT.fp4_gemm_grouped(aq, asq, B, BS)
want = torch.stack([KT.fp4_gemm(aq[e], asq[e], b_l[e].view(torch.float4_e2m1fn_x2), bs_l[e], E8)
                    for e in range(Eg)])
check("grouped E=6 per-expert A", got, want)

print("FAILED: " + ", ".join(fail) if fail else "ALL PASS")
sys.exit(1 if fail else 0)
