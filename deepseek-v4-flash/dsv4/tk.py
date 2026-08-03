"""Triton replacements for the hot torch stand-ins in kernels_torch.

Profiling one decode step showed 26 460 kernel launches and 96 ms of pure
cudaLaunchKernel: the torch versions are correct but each is a dozen elementwise
passes.  The two worst offenders:

  * hc_split_sinkhorn -- 20 Sinkhorn iterations x 4 ops x 43 layers = ~3 400 launches
    on 4x4 matrices (aten::sum 3 677 calls, aten::div 3 870 calls);
  * fp8_gemm -- materialises the dequantised weight with repeat_interleave + mul
    before every matmul (53 ms of aten::mul, 17 ms of repeat_interleave per token).

Each kernel here keeps kernels_torch's exact semantics, including that the stored
scale is a DIVISOR (quantise x/s, dequantise q*s) and that e8m0 encodes 2**(bits-127).
"""
import torch
import triton
import triton.language as tl

from kernels_torch import fp4_act_quant as _ref_fp4_act_quant  # bound before the swap-in

FP8_MAX = 448.0
FP4_MAX = 6.0


# --------------------------------------------------------------------------- sinkhorn
@triton.jit
def _hc_kernel(MIX, SCALE, BASE, PRE, POST, COMB,
               HC: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, HC)
    row = MIX + r * ((2 + HC) * HC)
    s0 = tl.load(SCALE + 0).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)

    pre = tl.sigmoid(tl.load(row + i).to(tl.float32) * s0
                     + tl.load(BASE + i).to(tl.float32)) + EPS
    post = 2.0 * tl.sigmoid(tl.load(row + HC + i).to(tl.float32) * s1
                            + tl.load(BASE + HC + i).to(tl.float32))

    off = 2 * HC + i[:, None] * HC + tl.arange(0, HC)[None, :]
    c = tl.load(row + off).to(tl.float32) * s2 + tl.load(BASE + off).to(tl.float32)
    c = tl.exp(c - tl.max(c, axis=1)[:, None])
    c = c / tl.sum(c, axis=1)[:, None] + EPS               # softmax(-1) + eps
    c = c / (tl.sum(c, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, axis=1)[:, None] + EPS)
        c = c / (tl.sum(c, axis=0)[None, :] + EPS)

    tl.store(PRE + r * HC + i, pre.to(PRE.dtype.element_ty))
    tl.store(POST + r * HC + i, post.to(POST.dtype.element_ty))
    tl.store(COMB + r * HC * HC + i[:, None] * HC + tl.arange(0, HC)[None, :],
             c.to(COMB.dtype.element_ty))


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    b, s, _ = mixes.size()
    hc, R = hc_mult, b * s
    dt = mixes.dtype
    m = mixes.contiguous()
    pre = torch.empty((R, hc), dtype=dt, device=m.device)
    post = torch.empty((R, hc), dtype=dt, device=m.device)
    comb = torch.empty((R, hc, hc), dtype=dt, device=m.device)
    _hc_kernel[(R,)](m, hc_scale.contiguous().float(), hc_base.contiguous().float(),
                     pre, post, comb,
                     HC=hc, ITERS=sinkhorn_iters, EPS=eps, num_warps=1)
    return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


# ------------------------------------------------------------------------ act_quant
@triton.jit
def _quant_kernel(X, Q, S, G, sr,
                  BLOCK: tl.constexpr, POW2: tl.constexpr, FP4: tl.constexpr,
                  INPLACE: tl.constexpr, E8M0: tl.constexpr, MAXV: tl.constexpr,
                  FLOOR: tl.constexpr):
    pid = tl.program_id(0)
    r, g = pid // G, pid % G
    off = r * sr + g * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), FLOOR)
    if POW2:
        s = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / MAXV))))
    else:
        s = amax * (1.0 / MAXV)
    v = x / s
    v = tl.minimum(tl.maximum(v, -MAXV), MAXV)
    if FP4:
        # round to the e2m1 grid: values 0 .5 1 1.5 2 3 4 6
        a = tl.abs(v)
        # boundaries are inclusive-upper, matching torch.bucketize(right=False):
        # after dividing by a power-of-two scale, exact midpoint hits are common.
        q = tl.where(a <= 0.25, 0.0,
            tl.where(a <= 0.75, 0.5,
            tl.where(a <= 1.25, 1.0,
            tl.where(a <= 1.75, 1.5,
            tl.where(a <= 2.5, 2.0,
            tl.where(a <= 3.5, 3.0,
            tl.where(a <= 5.0, 4.0, 6.0)))))))
        q = tl.where(v < 0, -q, q)
    else:
        q = v.to(tl.float8e4nv).to(tl.float32)
    if INPLACE:
        tl.store(X + off, (q * s).to(X.dtype.element_ty))
    else:
        tl.store(Q + r * (G * BLOCK) + g * BLOCK + tl.arange(0, BLOCK),
                 q.to(Q.dtype.element_ty))
        if E8M0:
            tl.store(S + r * G + g, (tl.log2(s) + 127.0).to(tl.uint8))
        else:
            tl.store(S + r * G + g, s.to(S.dtype.element_ty))


def _rowinfo(x):
    """(rows, row stride) for a tensor whose last dim may be a slice of a bigger one."""
    assert x.stride(-1) == 1, "last dim must be unit-stride"
    if x.dim() == 1:
        return 1, 0
    sr = x.stride(-2)
    n = 1
    for i in range(x.dim() - 1):
        n *= x.size(i)
        if i < x.dim() - 2:
            assert x.stride(i) == x.size(i + 1) * x.stride(i + 1), "irregular leading strides"
    return n, sr


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    R, sr = _rowinfo(x)
    G = N // block_size
    pow2 = scale_fmt is not None
    if inplace:
        _quant_kernel[(R * G,)](x, x, x, G, sr, BLOCK=block_size, POW2=pow2, FP4=False,
                                INPLACE=True, E8M0=False, MAXV=FP8_MAX, FLOOR=1e-4,
                                num_warps=4)
        return x
    e8m0 = scale_dtype == torch.float8_e8m0fnu
    q = torch.empty((R, N), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((R, G), dtype=torch.uint8 if e8m0 else scale_dtype, device=x.device)
    _quant_kernel[(R * G,)](x, q, s, G, sr, BLOCK=block_size, POW2=pow2, FP4=False,
                            INPLACE=False, E8M0=e8m0, MAXV=FP8_MAX, FLOOR=1e-4, num_warps=4)
    shape = tuple(x.shape[:-1])
    return (q.view(*shape, N),
            (s.view(torch.float8_e8m0fnu) if e8m0 else s).view(*shape, G))


def fp4_act_quant(x, block_size=32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    R, sr = _rowinfo(x)
    G = N // block_size
    if inplace:
        _quant_kernel[(R * G,)](x, x, x, G, sr, BLOCK=block_size, POW2=True, FP4=True,
                                INPLACE=True, E8M0=False, MAXV=FP4_MAX,
                                FLOOR=FP4_MAX * 2.0 ** -126, num_warps=2)
        return x
    return _ref_fp4_act_quant(x, block_size, inplace)   # packing path is not hot


# ------------------------------------------------------------------------- fp8_gemm
@triton.jit
def _fp8_gemm_kernel(A, AS, B, BS, C, M, N, K,
                     sa_m, sa_k, sas_m, sas_k, sb_n, sb_k, sbs_n, sbs_k, sc_m, sc_n,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * sa_m + (k0 + offs_k)[None, :] * sa_k,
                    mask=mask_m[:, None], other=0.0).to(tl.float32)
        asc = tl.load(AS + offs_m * sas_m + (k0 // BLOCK_K) * sas_k, mask=mask_m, other=127)
        b = tl.load(B + offs_n[:, None] * sb_n + (k0 + offs_k)[None, :] * sb_k,
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        bsc = tl.load(BS + pid_n * sbs_n + (k0 // BLOCK_K) * sbs_k)
        a = a * tl.exp2(asc.to(tl.float32) - 127.0)[:, None]
        b = b * tl.exp2(bsc.to(tl.float32) - 127.0)
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)), acc)
    tl.store(C + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n,
             acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _e8m0(t):
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8)
    if t.dtype == torch.uint8:
        return t
    return (torch.log2(t.float()).round() + 127).clamp(0, 254).to(torch.uint8)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T, B scaled per 128x128 block."""
    shape = tuple(a.shape[:-1])
    K, N = a.size(-1), b.size(0)
    a2 = a.reshape(-1, K)
    as2 = _e8m0(a_s).reshape(-1, a_s.size(-1)).contiguous()
    bs = _e8m0(b_s).contiguous()
    M = a2.size(0)
    c = torch.empty((M, N), dtype=torch.get_default_dtype(), device=a.device)
    grid = (triton.cdiv(N, 128), triton.cdiv(M, 64))
    _fp8_gemm_kernel[grid](a2, as2, b, bs, c, M, N, K,
                           a2.stride(0), a2.stride(1), as2.stride(0), as2.stride(1),
                           b.stride(0), b.stride(1), bs.stride(0), bs.stride(1),
                           c.stride(0), c.stride(1),
                           BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=3)
    return c.view(*shape, N)
