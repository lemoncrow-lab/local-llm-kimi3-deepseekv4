"""Fused grouped fp4 (e2m1) GEMM for DeepSeek-V4-Flash's routed experts.

The torch stand-in in kernels_torch.fp4_gemm is correct but decodes the whole weight
to fp32 first: for one expert that is 3 x 32 MiB of materialised fp32 plus a dense
matmul against M=1.  Per token (258 experts) it moves ~24 GiB through VRAM and costs
~0.4 s.  This kernel reads the fp4 bytes straight out of global memory, unpacks and
scales them in registers, and never materialises a dequantised weight -- 3.2 GiB per
token, which is what the arithmetic actually requires.

It is also *grouped*: the E experts a token routes to are one launch, not E launches.
During decode every expert of a layer sees the same activation row, so A is broadcast
across the group with stride 0.

Layout, matching the checkpoint:
    a    [E, M, K]      float8_e4m3fn   (or [1, M, K], broadcast)
    a_s  [E, M, K//128] float8_e8m0fnu  scale is a DIVISOR at quantisation time,
    b    [E, N, K//2]   uint8           so dequantisation multiplies
    b_s  [E, N, K//32]  float8_e8m0fnu
    ->   [E, M, N]      bfloat16

e2m1 nibble -> value: sign = bit3, exp = bits 2:1, mantissa = bit0; the low nibble of
each byte is the *even* element (verified against convert.py's FP4_TABLE).
"""
import torch
import triton
import triton.language as tl

BLOCK_K = 128  # one fp8 activation-scale group; also 4 fp4 weight-scale groups


@triton.jit
def _grouped_fp4_gemm(
    A, AS, B, BS, C, IDX,
    M, N, K,
    sa_e, sa_m, sa_k,
    sas_e, sas_m, sas_k,
    sb_e, sb_n, sb_k,
    sbs_e, sbs_n, sbs_k,
    sc_e, sc_m, sc_n,
    HAS_IDX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.program_id(2)
    # the weights live in a cache pool; IDX maps group position -> pool slot
    b_e = tl.load(IDX + e).to(tl.int64) if HAS_IDX else e

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_b = tl.arange(0, BLOCK_K // 2)
    mask_m = offs_m < M
    mask_n = offs_n < N

    A += e * sa_e
    AS += e * sas_e
    B += b_e * sb_e
    BS += b_e * sbs_e
    C += e * sc_e

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * sa_m + (k0 + offs_k)[None, :] * sa_k,
                    mask=mask_m[:, None], other=0.0).to(tl.float32)
        asc = tl.load(AS + offs_m * sas_m + (k0 // BLOCK_K) * sas_k, mask=mask_m, other=127)
        a = a * tl.exp2(asc.to(tl.float32) - 127.0)[:, None]

        raw = tl.load(B + offs_n[:, None] * sb_n + (k0 // 2 + offs_b)[None, :] * sb_k,
                      mask=mask_n[:, None], other=0)
        lo = (raw & 0xF).to(tl.int32)
        hi = ((raw >> 4) & 0xF).to(tl.int32)
        code = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
        exp = (code >> 1) & 3
        man = (code & 1).to(tl.float32)
        val = tl.where(exp == 0, man * 0.5,
                       (1.0 + man * 0.5) * tl.exp2((exp - 1).to(tl.float32)))
        val = tl.where((code >> 3) & 1 == 1, -val, val)

        bsc = tl.load(BS + offs_n[:, None] * sbs_n + ((k0 + offs_k) // 32)[None, :] * sbs_k,
                      mask=mask_n[:, None], other=127)
        w = val * tl.exp2(bsc.to(tl.float32) - 127.0)

        # fp8 e4m3 (4 significant bits) x fp4 e2m1 (2) fits bf16's 8 exactly:
        # the bf16 cast is lossless and buys tensor cores.
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), acc)

    tl.store(C + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n,
             acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _e8m0(t):
    """View an e8m0 scale tensor as the raw uint8 exponent the kernel expects."""
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8)
    if t.dtype == torch.uint8:
        return t
    # float scales: they are always powers of two here, so recover the exponent
    return (torch.log2(t.float()).round() + 127).clamp(0, 254).to(torch.uint8)


def fp4_gemm_grouped(a, a_s, b, b_s, out=None, idx=None):
    """[E,M,K] fp8 x [*,N,K] fp4 -> [E,M,N] bf16.

    a/a_s with E==1 broadcast across the group.  `idx` (int32 [E]) selects which slots
    of the b/b_s pools the group maps to; without it b is taken as [E,N,K] directly.
    """
    N = b.shape[1]
    E = b.shape[0] if idx is None else idx.numel()
    M, K = a.shape[-2], a.shape[-1]
    assert b.shape[2] * 2 == K, (b.shape, K)
    assert K % BLOCK_K == 0
    a, a_s, b_s = a.contiguous(), _e8m0(a_s).contiguous(), _e8m0(b_s)
    # b/b_s are views into the cache pool: strided on the expert axis, never copied
    assert b.stride(-1) == 1 and b_s.stride(-1) == 1
    if out is None:
        out = torch.empty((E, M, N), dtype=torch.bfloat16, device=b.device)

    ae = 0 if a.shape[0] == 1 else a.stride(0)
    ase = 0 if a_s.shape[0] == 1 else a_s.stride(0)
    block_n = 128 if N >= 128 else 32
    block_m = 16
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m), E)
    _grouped_fp4_gemm[grid](
        a, a_s, b, b_s, out, idx if idx is not None else a,
        M, N, K,
        ae, a.stride(-2), a.stride(-1),
        ase, a_s.stride(-2), a_s.stride(-1),
        b.stride(0), b.stride(1), b.stride(2),
        b_s.stride(0), b_s.stride(1), b_s.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_IDX=idx is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """Drop-in for kernels_torch.fp4_gemm: [M,K] x [N,K] -> [M,N]."""
    raw = b.view(torch.uint8) if b.dtype != torch.uint8 else b
    c = fp4_gemm_grouped(a.unsqueeze(0), a_s.unsqueeze(0), raw.unsqueeze(0), b_s.unsqueeze(0))
    return c.squeeze(0).to(torch.get_default_dtype())
