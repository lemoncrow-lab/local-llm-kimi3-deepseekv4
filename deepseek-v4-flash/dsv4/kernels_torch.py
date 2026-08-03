"""Pure-PyTorch stand-ins for DeepSeek-V4-Flash's tilelang kernels.

The checkpoint ships tilelang kernels authored for Blackwell/Hopper.  On this box
(RTX 4090, sm_89) `sparse_attn` asks for 141 312 B of dynamic shared memory against
Ada's 101 376 B cap, and `fp4_gemm` is tiled for M=32 and sustains 12 GB/s -- so the
reference path cannot run at all and would be slow if it did.

These replacements follow `inference/kernel.py` semantics exactly (see the line refs
in each docstring); they are correct first and vectorised second, with no tilelang,
no nvcc and no shared-memory ceiling.

Quantisation convention, taken from act_quant_kernel: the stored scale is a DIVISOR --
quantise with x/s, dequantise with q*s.
"""
import torch

FP8_MAX = 448.0
FP4_MAX = 6.0

# e2m1 code -> value, index = 4-bit nibble (sign in bit 3)
_FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# midpoints between consecutive magnitudes, for round-to-nearest quantisation
_FP4_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

_cache = {}


def _const(name, values, device, dtype=torch.float32):
    key = (name, device, dtype)
    if key not in _cache:
        _cache[key] = torch.tensor(values, device=device, dtype=dtype)
    return _cache[key]


def _round_scale(amax, max_inv):
    """fast_round_scale: 2 ** ceil(log2(amax * max_inv)) (kernel.py L36)."""
    return torch.exp2(torch.ceil(torch.log2(amax * max_inv)))


def _expand_scale(s, groups, dim_size):
    """[..., G] scales -> [..., G*group] covering dim_size columns."""
    out = s.repeat_interleave(groups, dim=-1)
    return out[..., :dim_size]


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise fp8 e4m3 quantisation along the last dim (kernel.py L41-L127)."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    g = z.float().unflatten(-1, (N // block_size, block_size))
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale(amax, 1.0 / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = (g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2).contiguous(), s.squeeze(-1).to(scale_dtype).contiguous()


def _fp4_encode(v):
    """float tensor in [-6, 6] -> 4-bit e2m1 codes (uint8)."""
    mag = v.abs()
    code = torch.bucketize(mag, _const("mid", _FP4_MID, v.device)).to(torch.uint8)
    return code | (torch.signbit(v).to(torch.uint8) << 3)


def _fp4_decode(codes):
    """4-bit e2m1 codes -> float values."""
    return _const("lut", _FP4_LUT, codes.device)[codes.long()]


def fp4_act_quant(x, block_size=32, inplace=False):
    """Block-wise fp4 e2m1 quantisation, power-of-2 scales (kernel.py L129-L201)."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    g = z.float().unflatten(-1, (N // block_size, block_size))
    amax = g.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2.0 ** -126)
    s = _round_scale(amax, 1.0 / FP4_MAX)
    v = (g / s).clamp(-FP4_MAX, FP4_MAX)
    codes = _fp4_encode(v)
    if inplace:
        x.copy_((_fp4_decode(codes) * s).flatten(-2).to(x.dtype))
        return x
    lo, hi = codes.flatten(-2)[..., 0::2], codes.flatten(-2)[..., 1::2]
    packed = (lo | (hi << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    return packed, s.squeeze(-1).to(torch.float8_e8m0fnu).contiguous()


def fp4_unpack(b):
    """[N, K//2] float4_e2m1fn_x2 -> [N, K] float32.  Low nibble is the even element."""
    raw = b.view(torch.uint8)
    both = torch.stack((raw & 0xF, raw >> 4), dim=-1).flatten(-2)
    return _fp4_decode(both)


def _dequant_a(a, a_s):
    """fp8 acts with per-128 scales along K -> float32."""
    K = a.size(-1)
    return a.float() * _expand_scale(a_s.float(), K // a_s.size(-1), K)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T; B scaled per 128x128 block (kernel.py L204-L274)."""
    K, N = a.size(-1), b.size(0)
    bs = b_s.float()
    w = b.float() * _expand_scale(bs, 128, K).repeat_interleave(128, dim=0)[:N]
    c = _dequant_a(a, a_s) @ w.T
    return c.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; B scaled per 32 along K (kernel.py L442-L536)."""
    w = fp4_unpack(b)
    K = w.size(-1)
    w = w * _expand_scale(b_s.float(), K // b_s.size(-1), K)
    c = _dequant_a(a, a_s) @ w.T
    return c.to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=128):
    """Gathered-index attention with a sink logit (kernel.py L277-L369).

    kv is the MLA latent and serves as both keys and values.  topk_idxs entries of -1
    are masked out.  attn_sink adds exp(sink - rowmax) to the denominator only: it
    absorbs probability mass but contributes no value.
    """
    b, s, h, d = q.size()
    out = q.new_empty(b, s, h, d)
    sink = attn_sink.float().view(1, 1, h)
    for lo in range(0, s, chunk):
        hi = min(lo + chunk, s)
        idx = topk_idxs[:, lo:hi].to(q.device).long()          # [b, t, topk]
        valid = idx >= 0
        bi = torch.arange(b, device=q.device).view(b, 1, 1)
        g = kv[bi, idx.clamp_min(0)].float()                   # [b, t, topk, d]
        sc = torch.einsum("bthd,btkd->bthk", q[:, lo:hi].float(), g) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        m = sc.amax(-1)                                        # [b, t, h]
        p = torch.exp(sc - m.unsqueeze(-1))
        denom = p.sum(-1) + torch.exp(sink - m)
        o = torch.einsum("bthk,btkd->bthd", p, g) / denom.unsqueeze(-1)
        out[:, lo:hi] = o.to(q.dtype)
    return out


def hadamard_transform(x, scale=1.0):
    """Walsh-Hadamard transform along the last dim, Sylvester ordering, times `scale`.

    Stands in for the `fast_hadamard_transform` CUDA extension (sdist-only, needs nvcc),
    which model.rotate_activation calls before fp8/fp4 quantisation.  The sizes in play
    are 128 and 512, so an explicit matmul is cheap.
    """
    n = x.size(-1)
    assert n & (n - 1) == 0, f"Hadamard needs a power of two, got {n}"
    key = ("hadamard", n, x.device, x.dtype)
    if key not in _cache:
        h = torch.ones(1, 1, device=x.device, dtype=torch.float32)
        while h.size(0) < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _cache[key] = h.to(x.dtype)
    return (x @ _cache[key]) * scale


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Hyper-Connection pre/post/combination weights (kernel.py L372-L439)."""
    b, s, _ = mixes.size()
    hc = hc_mult
    m = mixes.view(-1, (2 + hc) * hc).float()
    pre = torch.sigmoid(m[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[:, 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return (pre.view(b, s, hc).to(mixes.dtype),
            post.view(b, s, hc).to(mixes.dtype),
            comb.view(b, s, hc, hc).to(mixes.dtype))
