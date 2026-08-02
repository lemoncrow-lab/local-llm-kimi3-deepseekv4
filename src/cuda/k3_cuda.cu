/* SPDX-License-Identifier: Apache-2.0 */
/* New 2026-08: optional CUDA/NVRTC backend and compressed cache. See MODIFICATIONS.md. */
/* Optional CUDA backend, with kernels compiled by NVRTC at process startup. */
#include <cuda.h>
#include <nvrtc.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"

static const char *KERNEL_SOURCE = R"CUDA(
extern "C" {

/* Per-group scales are stored as bf16: the top 16 bits of the f32, round to nearest
 * even. A scale needs range, not mantissa, and at group 128 this is 0.85 GB of VRAM
 * across the trunk -- the difference between fitting on the card and not. */
__device__ __forceinline__ float bf16_scale(unsigned short h)
{
    return __uint_as_float((unsigned int)h << 16);
}

__device__ __forceinline__ unsigned short pack_bf16(float v)
{
    const unsigned int u = __float_as_uint(v);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float warp_sum(float v)
{
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}

__device__ __forceinline__ void finish_row(float v, float *out, int row)
{
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const int nwarp = (blockDim.x + 31) >> 5;
        v = lane < nwarp ? warp_sums[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) out[row] = v;
    }
}

__global__ void bf16_gemv(float *out, const float *x,
                          const unsigned short *weights, int in, int rows)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short *w =
        weights + (unsigned long long)row * (unsigned long long)in;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in; i += blockDim.x) {
        const float wf = __uint_as_float((unsigned int)w[i] << 16);
        sum = fmaf(wf, x[i], sum);
    }
    finish_row(sum, out, row);
}

__device__ __constant__ float E2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__global__ void mxfp4_gemv(float *out, const float *x,
                           const unsigned char *packed,
                           const unsigned char *scales,
                           int in, int rows, int group)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int pcols = in >> 1;
    const int ngrp = (in + group - 1) / group;
    const unsigned char *pr =
        packed + (unsigned long long)row * (unsigned long long)pcols;
    const unsigned char *sr =
        scales + (unsigned long long)row * (unsigned long long)ngrp;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in; i += blockDim.x) {
        const unsigned char sb = sr[i / group];
        if (sb == 255) continue;
        const unsigned char byte = pr[i >> 1];
        const unsigned char nib = (i & 1) ? (byte >> 4) : (byte & 15);
        const float scale = ldexpf(1.0f, (int)sb - 127);
        sum = fmaf(E2M1[nib] * scale, x[i], sum);
    }
    finish_row(sum, out, row);
}

__device__ __forceinline__ float warp_max(float v)
{
    for (int d = 16; d > 0; d >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, d));
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float *shared)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) shared[31] = v;
    __syncthreads();
    return shared[31];
}

__device__ __forceinline__ float block_reduce_sum(float v, float *shared)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) shared[31] = v;
    __syncthreads();
    return shared[31];
}

/* Per-group quantisation scale.
 *
 * absmax/qmax is the obvious choice and it is not the best one: it sizes the step so
 * that the single largest element of the group is exactly representable, which lets
 * one outlier dictate the resolution of the other 127. With opt set, a few finer
 * steps are tried and the one with the lowest squared reconstruction error over the
 * whole group wins -- a little clipping on the extreme bought for resolution
 * everywhere else. Same trade as llama.cpp's make_qx_quants, and it is worth more at
 * three bits than at four, which is exactly where the mixed cache spends them.
 * K3_CUDA_QOPT=0 restores the plain absmax scale. */
__global__ void block_scale(const unsigned short *weights, unsigned short *scales,
                            int in, int rows, int group, int qmax, int opt)
{
    const int ngrp = (in + group - 1) / group;
    const int si = blockIdx.x;
    if (si >= rows * ngrp) return;
    const int row = si / ngrp;
    const int gi = si - row * ngrp;
    const int begin = gi * group;
    const int end = begin + group < in ? begin + group : in;
    const unsigned short *w =
        weights + (unsigned long long)row * (unsigned long long)in;
    __shared__ float shared[32];

    float m = 0.0f;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float v = __uint_as_float((unsigned int)w[i] << 16);
        m = fmaxf(m, fabsf(v));
    }
    m = block_reduce_max(m, shared);
    if (m <= 0.0f) {
        if (threadIdx.x == 0) scales[si] = pack_bf16(1.0f);
        return;
    }
    if (!opt) {
        if (threadIdx.x == 0) scales[si] = pack_bf16(m / (float)qmax);
        return;
    }

    float best_scale = m / (float)qmax;
    float best_err = -1.0f;
    for (int trial = 0; trial < 8; trial++) {
        const float sc = m / ((float)qmax + 0.08f * (float)trial);
        const float inv = 1.0f / sc;
        float err = 0.0f;
        for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
            const float v = __uint_as_float((unsigned int)w[i] << 16);
            float q = rintf(v * inv);
            q = fminf(fmaxf(q, -(float)qmax), (float)qmax);   /* as the packers clamp */
            const float d = v - q * sc;
            err += d * d;
        }
        err = block_reduce_sum(err, shared);
        if (best_err < 0.0f || err < best_err) { best_err = err; best_scale = sc; }
    }
    if (threadIdx.x == 0) scales[si] = pack_bf16(best_scale);
}

__global__ void q3_pack(const unsigned short *weights, const unsigned short *scales,
                        unsigned char *packed, int in, int rows, int group)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int groups = in >> 3;
    const int row_bytes = groups * 3;
    const unsigned short *w =
        weights + (unsigned long long)row * (unsigned long long)in;
    unsigned char *p =
        packed + (unsigned long long)row * (unsigned long long)row_bytes;
    const int ngrp = (in + group - 1) / group;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const float inv = 1.0f / bf16_scale(scales[row * ngrp + (g * 8) / group]);
        unsigned int bits = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float v = __uint_as_float((unsigned int)w[g * 8 + j] << 16);
            int q = (int)rintf(v * inv);
            q = q < -3 ? -3 : (q > 3 ? 3 : q);
            bits |= (unsigned int)(q + 3) << (3 * j);
        }
        p[g * 3 + 0] = (unsigned char)bits;
        p[g * 3 + 1] = (unsigned char)(bits >> 8);
        p[g * 3 + 2] = (unsigned char)(bits >> 16);
    }
}

__global__ void q3_gemv(float *out, const float *x,
                        const unsigned char *packed, const unsigned short *scales,
                        int in, int rows, int group)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int groups = in >> 3;
    const int row_bytes = groups * 3;
    const unsigned char *p =
        packed + (unsigned long long)row * (unsigned long long)row_bytes;
    float sum = 0.0f;
    const int ngrp = (in + group - 1) / group;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const float scale = bf16_scale(scales[row * ngrp + (g * 8) / group]);
        const unsigned int bits = (unsigned int)p[g * 3]
                                | ((unsigned int)p[g * 3 + 1] << 8)
                                | ((unsigned int)p[g * 3 + 2] << 16);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int q = (int)((bits >> (3 * j)) & 7u) - 3;
            sum = fmaf((float)q * scale, x[g * 8 + j], sum);
        }
    }
    finish_row(sum, out, row);
}

__global__ void q4_pack(const unsigned short *weights, const unsigned short *scales,
                        unsigned char *packed, int in, int rows, int group)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int pairs = in >> 1;
    const int ngrp = (in + group - 1) / group;
    const unsigned short *w =
        weights + (unsigned long long)row * (unsigned long long)in;
    unsigned char *p =
        packed + (unsigned long long)row * (unsigned long long)pairs;
    for (int g = threadIdx.x; g < pairs; g += blockDim.x) {
        const float inv = 1.0f / bf16_scale(scales[row * ngrp + (g * 2) / group]);
        int a = (int)rintf(__uint_as_float((unsigned int)w[g * 2] << 16) * inv);
        int b = (int)rintf(__uint_as_float((unsigned int)w[g * 2 + 1] << 16) * inv);
        a = a < -7 ? -7 : (a > 7 ? 7 : a);
        b = b < -7 ? -7 : (b > 7 ? 7 : b);
        p[g] = (unsigned char)((a + 7) | ((b + 7) << 4));
    }
}

/* The router gate is the one matrix the binder hands over already widened to f32
 * (k3_router carries its own inline matmul). Packing it needs a float-source twin of
 * the scale and pack kernels; the GEMV is shared, because what comes out the other
 * side is the same packed format either way. */
__global__ void block_scale_f32(const float *weights, unsigned short *scales,
                                int in, int rows, int group, int qmax, int opt)
{
    const int ngrp = (in + group - 1) / group;
    const int si = blockIdx.x;
    if (si >= rows * ngrp) return;
    const int row = si / ngrp;
    const int gi = si - row * ngrp;
    const int begin = gi * group;
    const int end = begin + group < in ? begin + group : in;
    const float *w = weights + (unsigned long long)row * (unsigned long long)in;
    __shared__ float shared[32];

    float m = 0.0f;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
        m = fmaxf(m, fabsf(w[i]));
    m = block_reduce_max(m, shared);
    if (m <= 0.0f) {
        if (threadIdx.x == 0) scales[si] = pack_bf16(1.0f);
        return;
    }
    if (!opt) {
        if (threadIdx.x == 0) scales[si] = pack_bf16(m / (float)qmax);
        return;
    }
    float best_scale = m / (float)qmax;
    float best_err = -1.0f;
    for (int trial = 0; trial < 8; trial++) {
        const float sc = m / ((float)qmax + 0.08f * (float)trial);
        const float inv = 1.0f / sc;
        float err = 0.0f;
        for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
            float q = rintf(w[i] * inv);
            q = fminf(fmaxf(q, -(float)qmax), (float)qmax);
            const float d = w[i] - q * sc;
            err += d * d;
        }
        err = block_reduce_sum(err, shared);
        if (best_err < 0.0f || err < best_err) { best_err = err; best_scale = sc; }
    }
    if (threadIdx.x == 0) scales[si] = pack_bf16(best_scale);
}

__global__ void q4_pack_f32(const float *weights, const unsigned short *scales,
                            unsigned char *packed, int in, int rows, int group)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int pairs = in >> 1;
    const int ngrp = (in + group - 1) / group;
    const float *w = weights + (unsigned long long)row * (unsigned long long)in;
    unsigned char *p = packed + (unsigned long long)row * (unsigned long long)pairs;
    for (int g = threadIdx.x; g < pairs; g += blockDim.x) {
        const float inv = 1.0f / bf16_scale(scales[row * ngrp + (g * 2) / group]);
        int a = (int)rintf(w[g * 2] * inv);
        int b = (int)rintf(w[g * 2 + 1] * inv);
        a = a < -7 ? -7 : (a > 7 ? 7 : a);
        b = b < -7 ? -7 : (b > 7 ? 7 : b);
        p[g] = (unsigned char)((a + 7) | ((b + 7) << 4));
    }
}

__global__ void q4_gemv(float *out, const float *x,
                        const unsigned char *packed, const unsigned short *scales,
                        int in, int rows, int group)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int pairs = in >> 1;
    const int ngrp = (in + group - 1) / group;
    const unsigned char *p =
        packed + (unsigned long long)row * (unsigned long long)pairs;
    float sum = 0.0f;
    for (int g = threadIdx.x; g < pairs; g += blockDim.x) {
        const unsigned char q = p[g];
        const float scale = bf16_scale(scales[row * ngrp + (g * 2) / group]);
        sum = fmaf((float)((int)(q & 15) - 7) * scale, x[g * 2], sum);
        sum = fmaf((float)((int)(q >> 4) - 7) * scale, x[g * 2 + 1], sum);
    }
    finish_row(sum, out, row);
}

}
)CUDA";

typedef struct {
    const void *host;
    size_t bytes;
    int scope;
    int bits;            /* 0 = plain BF16, else 3 or 4 packed bits per weight  */
    int src_f32;         /* source matrix was f32, not bf16                     */
    int on_host;         /* pinned-RAM overflow tier: re-uploaded every token   */
    int in, rows;
    CUdeviceptr device;
    CUdeviceptr scales;
    void *host_packed;
    void *host_scales;
    size_t packed_bytes;
    size_t scale_bytes;
    int host_pinned;
} K3CudaCacheEntry;

typedef struct {
    int enabled;
    int device;
    char name[256];
    CUcontext context;
    CUstream stream;
    CUmodule module;
    CUfunction bf16_fn;
    CUfunction mxfp4_fn;
    CUfunction block_scale_fn;
    CUfunction q3_pack_fn;
    CUfunction q3_fn;
    CUfunction q4_pack_fn;
    CUfunction q4_fn;
    CUfunction block_scale_f32_fn;
    CUfunction q4_pack_f32_fn;
    CUdeviceptr weights;
    CUdeviceptr scales;
    CUdeviceptr x;
    CUdeviceptr y;
    CUdeviceptr qweights;
    CUdeviceptr qscales;
    size_t weights_cap;
    size_t scales_cap;
    size_t x_cap;
    size_t y_cap;
    size_t qweights_cap;
    size_t qscales_cap;
    size_t min_bytes;
    int cacheable;
    int cache_scope;
    int q3_cache;
    int q4_cache;
    int mix;
    int qopt;
    int cpu_lane;
    CUdeviceptr arena;
    size_t arena_cap;
    size_t arena_used;
    size_t q4_max_elems;
    size_t stage_bytes;
    unsigned long long downgrades;
    int q3_group;
    K3CudaCacheEntry *cache;
    size_t cache_len;
    size_t cache_cap;
    size_t cache_bytes;
    size_t cache_budget;
    size_t host_cache_bytes;
    size_t host_cache_budget;
    size_t host_pinned_bytes;
    unsigned long long cache_hits;
    unsigned long long cache_misses;
    unsigned long long compressed_fallbacks;
    unsigned long long bf16_calls;
    unsigned long long mxfp4_calls;
    unsigned long long h2d_bytes;
    unsigned long long d2h_bytes;
    double wall_seconds;
    /* Per-step profile, reset by k3_cuda_step_reset. Attribution is by call path, and
     * every path synchronises before returning, so the wall time lands on the right
     * row. Without the split, "backend wall" hides a 9.2 GB/token PCIe leg behind
     * thousands of cheap resident-weight launches. */
    double st_dev_s, st_host_s, st_bf16_s, st_mxfp4_s;
    unsigned long long st_dev_n, st_host_n, st_bf16_n, st_mxfp4_n;
    unsigned long long st_host_bytes, st_bf16_bytes, st_mxfp4_bytes;
} K3CudaState;

static K3CudaState g_cuda;

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int driver_error(CUresult e, const char *where)
{
    if (e == CUDA_SUCCESS) return 0;
    const char *name = NULL, *text = NULL;
    cuGetErrorName(e, &name);
    cuGetErrorString(e, &text);
    fprintf(stderr, "k3_cuda: %s: %s (%s)\n", where,
            text ? text : "unknown CUDA error", name ? name : "unknown");
    return -1;
}

static int reserve(CUdeviceptr *ptr, size_t *cap, size_t need)
{
    if (*cap >= need) return 0;
    if (*ptr && driver_error(cuMemFree(*ptr), "cuMemFree") != 0) return -1;
    *ptr = 0;
    *cap = 0;
    if (driver_error(cuMemAlloc(ptr, need), "cuMemAlloc") != 0) return -1;
    *cap = need;
    return 0;
}

/* One resident weight, in whatever format it was admitted as.
 *
 * A streamed trunk reuses host addresses for different layers, so entries are keyed by
 * (host pointer, byte count, cache scope) and the caller marks only immutable tensors
 * cacheable. Nothing is ever evicted: eviction under a cyclic scan is a treadmill. */
static K3CudaCacheEntry *cache_find(const void *host, size_t bytes)
{
    for (size_t i = 0; i < g_cuda.cache_len; i++) {
        K3CudaCacheEntry *e = &g_cuda.cache[i];
        if (e->scope == g_cuda.cache_scope && e->host == host && e->bytes == bytes)
            return e;
    }
    return NULL;
}

static K3CudaCacheEntry *cache_push(void)
{
    if (g_cuda.cache_len == g_cuda.cache_cap) {
        size_t cap = g_cuda.cache_cap ? g_cuda.cache_cap * 2 : 128;
        void *p = realloc(g_cuda.cache, cap * sizeof(*g_cuda.cache));
        if (!p) return NULL;
        g_cuda.cache = (K3CudaCacheEntry *)p;
        g_cuda.cache_cap = cap;
    }
    K3CudaCacheEntry *e = &g_cuda.cache[g_cuda.cache_len++];
    memset(e, 0, sizeof *e);
    return e;
}

/* ONE arena, bump-allocated.
 *
 * The cache holds ~2,400 tensors, and cuMemAlloc rounds every request up to a 2 MB
 * granule: a 200 KB scale vector costs 2 MB, and the rounding alone lost more than a
 * gigabyte -- enough to push the trunk back out of VRAM and into the PCIe tier, which
 * is the one thing this cache exists to prevent. A single allocation with a bump
 * pointer has no rounding, no fragmentation, and no failure mode halfway through a
 * warm-up. */
static int arena_reserve(size_t bytes)
{
    if (g_cuda.arena) return 0;
    if (driver_error(cuMemAlloc(&g_cuda.arena, bytes), "cuMemAlloc(arena)") != 0) {
        g_cuda.arena = 0;
        return -1;
    }
    g_cuda.arena_cap = bytes;
    g_cuda.arena_used = 0;
    return 0;
}

static int arena_take(size_t bytes, CUdeviceptr *out)
{
    const size_t align = 512;
    size_t base = (g_cuda.arena_used + align - 1) & ~(align - 1);
    if (base + bytes > g_cuda.arena_cap) return 1;
    *out = g_cuda.arena + base;
    g_cuda.arena_used = base + bytes;
    return 0;
}

/* Plain BF16 residency, used only when no packed format is selected. Kept because it
 * is the one path that changes no arithmetic at all, which makes it the reference the
 * packed formats are measured against. */
static int cached_bf16(const void *host, size_t bytes, CUdeviceptr *device,
                       int *needs_upload)
{
    *needs_upload = 0;
    if (!g_cuda.cacheable || g_cuda.cache_budget == 0 ||
        g_cuda.q3_cache || g_cuda.q4_cache || g_cuda.mix) return 1;

    K3CudaCacheEntry *e = cache_find(host, bytes);
    if (e && e->bits == 0) { *device = e->device; g_cuda.cache_hits++; return 0; }

    g_cuda.cache_misses++;
    CUdeviceptr ptr = 0;
    if (arena_take(bytes, &ptr) != 0) return 1;
    e = cache_push();
    if (!e) return -1;
    e->host = host;
    e->bytes = bytes;
    e->scope = g_cuda.cache_scope;
    e->bits = 0;
    e->device = ptr;
    e->packed_bytes = bytes;
    g_cuda.cache_bytes = g_cuda.arena_used;
    *device = ptr;
    *needs_upload = 1;
    return 0;
}

static size_t packed_size(int bits, int in, int rows)
{
    return bits == 3 ? (size_t)rows * (size_t)(in / 8) * 3
                     : (size_t)rows * (size_t)(in / 2);
}

static size_t scale_size(int in, int rows)
{
    const int group = g_cuda.q3_group;
    return (size_t)rows * (size_t)((in + group - 1) / group) * sizeof(unsigned short);
}

/* Which precision a matrix is admitted at, and the whole point of the mixed cache.
 *
 * At Q4 this trunk packs to 28.9 GB. A 24 GB card holds 20 of it, so 9.2 GB is
 * re-copied over PCIe every single token: half a second per token, more than every
 * other backend cost put together. At Q3 it packs to 22.1 GB and fits -- but three
 * bits everywhere is accuracy the small matrices need not have paid. So the large
 * matrices take Q3, everything at or under the threshold keeps Q4, and the threshold
 * is a knob because how much VRAM is free is a property of the machine, not the model.
 * lm_head is special-cased to Q4: it is the last thing between the model and the
 * argmax and only 0.6 GB packed. */
static int want_bits(int in, int rows)
{
    if (!g_cuda.mix) return g_cuda.q3_cache ? 3 : 4;
    if (rows >= 100000) return 4;                      /* lm_head */
    return ((size_t)in * (size_t)rows > g_cuda.q4_max_elems) ? 3 : 4;
}

/* Reserve a resident slot. Order of preference: the requested bits on the device,
 * then three bits on the device, then the pinned-host overflow tier. The middle step
 * is what keeps a run resident instead of tipping into per-token PCIe when the budget
 * falls a little short. */
static int cached_quant(const void *host, size_t source_bytes, int in, int rows,
                        K3CudaCacheEntry **entry, int *needs_quantize)
{
    *needs_quantize = 0;
    if (!(g_cuda.q3_cache || g_cuda.q4_cache || g_cuda.mix)) return 1;
    if (!g_cuda.cacheable || (in & 7)) {   /* the packers need whole bytes and octets */
        g_cuda.compressed_fallbacks++;
        return 1;
    }

    K3CudaCacheEntry *e = cache_find(host, source_bytes);
    if (e) { *entry = e; g_cuda.cache_hits++; return 0; }
    g_cuda.cache_misses++;

    const size_t sbytes = scale_size(in, rows);
    int bits = want_bits(in, rows);
    size_t pbytes = packed_size(bits, in, rows);
    const size_t room = g_cuda.arena_cap > g_cuda.arena_used
                      ? g_cuda.arena_cap - g_cuda.arena_used : 0;
    int on_host = 0;
    if (pbytes + sbytes > room && bits == 4) {         /* downgrade before offloading */
        bits = 3;
        pbytes = packed_size(bits, in, rows);
        g_cuda.downgrades++;
    }
    if (pbytes + sbytes > room) {
        if (pbytes + sbytes > g_cuda.host_cache_budget - g_cuda.host_cache_bytes) {
            g_cuda.compressed_fallbacks++;
            return 1;
        }
        on_host = 1;
    }

    e = cache_push();
    if (!e) return -1;

    CUdeviceptr packed = 0, scales = 0;
    void *host_packed = NULL, *host_scales = NULL;
    if (!on_host) {
        if (arena_take(pbytes, &packed) != 0 || arena_take(sbytes, &scales) != 0) {
            /* Cannot happen: room was checked above. Treated as a hard error rather
             * than silently producing a matmul against unallocated memory. */
            g_cuda.cache_len--;
            return -1;
        }
    } else {
        CUresult hp = cuMemHostAlloc(&host_packed, pbytes, CU_MEMHOSTALLOC_PORTABLE);
        CUresult hs = hp == CUDA_SUCCESS
            ? cuMemHostAlloc(&host_scales, sbytes, CU_MEMHOSTALLOC_PORTABLE)
            : CUDA_ERROR_OUT_OF_MEMORY;
        if (hp != CUDA_SUCCESS || hs != CUDA_SUCCESS) {
            if (hp == CUDA_SUCCESS) cuMemFreeHost(host_packed);
            if (hs == CUDA_SUCCESS) cuMemFreeHost(host_scales);
            host_packed = malloc(pbytes);
            host_scales = malloc(sbytes);
            if (!host_packed || !host_scales) {
                free(host_packed); free(host_scales);
                g_cuda.cache_len--;
                return -1;
            }
        }
    }

    e->host = host;
    e->bytes = source_bytes;
    e->scope = g_cuda.cache_scope;
    e->bits = bits;
    e->on_host = on_host;
    e->in = in;
    e->rows = rows;
    e->device = packed;
    e->scales = scales;
    e->host_packed = host_packed;
    e->host_scales = host_scales;
    e->packed_bytes = pbytes;
    e->scale_bytes = sbytes;
    e->host_pinned = on_host && host_packed &&
                     cuMemHostGetDevicePointer(&packed, host_packed, 0) == CUDA_SUCCESS;
    if (on_host) {
        g_cuda.host_cache_bytes += pbytes + sbytes;
        if (e->host_pinned) g_cuda.host_pinned_bytes += pbytes + sbytes;
    } else {
        g_cuda.cache_bytes = g_cuda.arena_used;
    }
    *entry = e;
    *needs_quantize = 1;
    return 0;
}

/* Quantise a BF16 matrix into its slot, ROW CHUNK BY ROW CHUNK.
 *
 * The staging buffer this avoids is not a rounding error. Staging a whole matrix means
 * holding the largest one (lm_head, 2.35 GB as BF16) in VRAM alongside the cache, and
 * reserving that 2.35 GB for the whole run although it is used only during warm-up.
 * Chunking caps it at K3_CUDA_STAGE_MB and hands the difference to the resident cache,
 * which is precisely the memory that decides whether the trunk fits. */
static int quantize_entry(K3CudaCacheEntry *e, const void *weights)
{
    const int in = e->in, rows = e->rows, bits = e->bits;
    int qgroup = g_cuda.q3_group;
    const int ngrp = (in + qgroup - 1) / qgroup;
    const size_t row_src = (size_t)in * (e->src_f32 ? sizeof(float) : sizeof(uint16_t));
    const size_t row_packed = bits == 3 ? (size_t)(in / 8) * 3 : (size_t)(in / 2);

    size_t chunk = g_cuda.stage_bytes / row_src;
    if (chunk == 0) chunk = 1;
    if (chunk > (size_t)rows) chunk = (size_t)rows;
    if (reserve(&g_cuda.weights, &g_cuda.weights_cap, chunk * row_src) != 0) return -1;

    /* The host tier packs through the same device path and is copied back afterwards.
     * There is no CPU packer, and writing one would be a second implementation of the
     * format to keep in agreement with the first. */
    CUdeviceptr dst_packed = e->device, dst_scales = e->scales;
    if (e->on_host) {
        if (reserve(&g_cuda.qweights, &g_cuda.qweights_cap, e->packed_bytes) != 0 ||
            reserve(&g_cuda.qscales, &g_cuda.qscales_cap, e->scale_bytes) != 0)
            return -1;
        dst_packed = g_cuda.qweights;
        dst_scales = g_cuda.qscales;
    }

    for (size_t r0 = 0; r0 < (size_t)rows; r0 += chunk) {
        int nr = (int)(((size_t)rows - r0) < chunk ? (size_t)rows - r0 : chunk);
        if (driver_error(cuMemcpyHtoDAsync(g_cuda.weights,
                                           (const char *)weights + r0 * row_src,
                                           (size_t)nr * row_src, g_cuda.stream),
                         "quantise source H2D") != 0)
            return -1;
        CUdeviceptr pk = dst_packed + r0 * row_packed;
        /* sizeof the SCALE type, not of float: scales are bf16, and a stale 4 here
         * silently mis-addresses every chunk after the first. */
        CUdeviceptr sc = dst_scales + r0 * (size_t)ngrp * sizeof(unsigned short);
        int qmax = bits == 3 ? 3 : 7;
        int qopt = g_cuda.qopt;
        void *scale_args[] = { &g_cuda.weights, &sc, (void *)&in, &nr, &qgroup,
                               &qmax, &qopt };
        if (driver_error(cuLaunchKernel(e->src_f32 ? g_cuda.block_scale_f32_fn
                                                   : g_cuda.block_scale_fn,
                                        (unsigned int)((size_t)nr * (size_t)ngrp), 1, 1,
                                        256, 1, 1, 0, g_cuda.stream, scale_args, NULL),
                         "block_scale launch") != 0)
            return -1;
        void *pack_args[] = { &g_cuda.weights, &sc, &pk, (void *)&in, &nr, &qgroup };
        if (driver_error(cuLaunchKernel(e->src_f32 ? g_cuda.q4_pack_f32_fn
                                        : (bits == 3 ? g_cuda.q3_pack_fn
                                                     : g_cuda.q4_pack_fn),
                                        (unsigned int)nr, 1, 1, 256, 1, 1,
                                        0, g_cuda.stream, pack_args, NULL),
                         "pack launch") != 0)
            return -1;
    }

    if (e->on_host) {
        if (driver_error(cuMemcpyDtoHAsync(e->host_packed, dst_packed,
                                           e->packed_bytes, g_cuda.stream),
                         "packed D2H") != 0 ||
            driver_error(cuMemcpyDtoHAsync(e->host_scales, dst_scales,
                                           e->scale_bytes, g_cuda.stream),
                         "scales D2H") != 0 ||
            driver_error(cuStreamSynchronize(g_cuda.stream), "host-tier sync") != 0)
            return -1;
        g_cuda.d2h_bytes += e->packed_bytes + e->scale_bytes;
    }
    g_cuda.h2d_bytes += (size_t)rows * row_src;
    return 0;
}

static int compile_kernels(void)
{
    nvrtcProgram program;
    nvrtcResult nr = nvrtcCreateProgram(&program, KERNEL_SOURCE, "k3_kernels.cu",
                                         0, NULL, NULL);
    if (nr != NVRTC_SUCCESS) {
        fprintf(stderr, "k3_cuda: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(nr));
        return -1;
    }
    const char *options[] = {
        "--gpu-architecture=compute_89",
        "--std=c++14"
    };
    nr = nvrtcCompileProgram(program, 2, options);
    if (nr != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            nvrtcGetProgramLog(program, log);
            log[log_size] = 0;
            fprintf(stderr, "k3_cuda: NVRTC compilation failed:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&program);
        return -1;
    }

    size_t ptx_size = 0;
    nvrtcGetPTXSize(program, &ptx_size);
    char *ptx = (char *)malloc(ptx_size);
    if (!ptx) {
        nvrtcDestroyProgram(&program);
        return -1;
    }
    nr = nvrtcGetPTX(program, ptx);
    nvrtcDestroyProgram(&program);
    if (nr != NVRTC_SUCCESS) {
        fprintf(stderr, "k3_cuda: nvrtcGetPTX: %s\n", nvrtcGetErrorString(nr));
        free(ptx);
        return -1;
    }

    const int failed =
        driver_error(cuModuleLoadData(&g_cuda.module, ptx), "cuModuleLoadData") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.bf16_fn, g_cuda.module, "bf16_gemv"),
                     "cuModuleGetFunction(bf16_gemv)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.mxfp4_fn, g_cuda.module, "mxfp4_gemv"),
                     "cuModuleGetFunction(mxfp4_gemv)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.block_scale_fn, g_cuda.module, "block_scale"),
                     "cuModuleGetFunction(block_scale)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.q3_pack_fn, g_cuda.module, "q3_pack"),
                     "cuModuleGetFunction(q3_pack)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.q3_fn, g_cuda.module, "q3_gemv"),
                     "cuModuleGetFunction(q3_gemv)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.q4_pack_fn, g_cuda.module, "q4_pack"),
                     "cuModuleGetFunction(q4_pack)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.q4_fn, g_cuda.module, "q4_gemv"),
                     "cuModuleGetFunction(q4_gemv)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.block_scale_f32_fn, g_cuda.module,
                                         "block_scale_f32"),
                     "cuModuleGetFunction(block_scale_f32)") != 0 ||
        driver_error(cuModuleGetFunction(&g_cuda.q4_pack_f32_fn, g_cuda.module,
                                         "q4_pack_f32"),
                     "cuModuleGetFunction(q4_pack_f32)") != 0;
    free(ptx);
    return failed ? -1 : 0;
}

extern "C" int k3_cuda_init(int device_index)
{
    if (g_cuda.enabled) return 0;
    if (driver_error(cuInit(0), "cuInit") != 0) return -1;
    CUdevice device;
    if (driver_error(cuDeviceGet(&device, device_index), "cuDeviceGet") != 0 ||
        driver_error(cuDeviceGetName(g_cuda.name, sizeof g_cuda.name, device),
                     "cuDeviceGetName") != 0 ||
        /* Spin, do not yield. Decode is ~1,500 launch-and-synchronise round trips per
         * token with a CPU-side graph between them, so scheduler wake-up latency is
         * paid 1,500 times a token; busy-waiting a core is the cheaper trade here. */
        driver_error(cuCtxCreate(&g_cuda.context, CU_CTX_SCHED_SPIN, device),
                     "cuCtxCreate") != 0 ||
        driver_error(cuStreamCreate(&g_cuda.stream, CU_STREAM_NON_BLOCKING),
                     "cuStreamCreate") != 0 ||
        compile_kernels() != 0)
        return -1;

    g_cuda.device = device_index;
    g_cuda.min_bytes = 1u << 20;
    const char *minimum = getenv("K3_CUDA_MIN_BYTES");
    if (minimum && *minimum) g_cuda.min_bytes = (size_t)strtoull(minimum, NULL, 10);

    const char *format = getenv("K3_CUDA_CACHE_FORMAT");
    g_cuda.q3_cache = format && !strcmp(format, "q3");
    g_cuda.q4_cache = format && !strcmp(format, "q4");
    g_cuda.mix      = format && !strcmp(format, "mix");
    if (format && *format && !g_cuda.q3_cache && !g_cuda.q4_cache && !g_cuda.mix &&
        strcmp(format, "bf16")) {
        fprintf(stderr, "k3_cuda: K3_CUDA_CACHE_FORMAT must be bf16, q3, q4 or mix\n");
        return -1;
    }
    /* Above this element count a matrix is packed at three bits instead of four. The
     * default is solved from this checkpoint's own shape mix: it puts the attention
     * projections and the shared experts (88M and 44M elements) at Q3 and keeps the
     * router gate, the latent projections and the low-rank pairs at Q4. */
    g_cuda.q4_max_elems = 26000000u;
    const char *q4max = getenv("K3_CUDA_Q4_MAX_ELEMS");
    if (q4max && *q4max) g_cuda.q4_max_elems = (size_t)strtoull(q4max, NULL, 10);
    /* On by default wherever a packed cache is NOT holding the whole trunk: that is
     * exactly the configuration where matrices spill, and spilling to the CPU beats
     * spilling to PCIe. */
    g_cuda.cpu_lane = !(g_cuda.q3_cache || g_cuda.q4_cache || g_cuda.mix);
    const char *lane_env = getenv("K3_CUDA_CPU_LANE");
    if (lane_env && *lane_env) g_cuda.cpu_lane = strcmp(lane_env, "0") != 0;
    g_cuda.qopt = 1;
    const char *qopt_env = getenv("K3_CUDA_QOPT");
    if (qopt_env && *qopt_env) g_cuda.qopt = atoi(qopt_env);
    g_cuda.stage_bytes = (size_t)256 << 20;
    const char *stage_env = getenv("K3_CUDA_STAGE_MB");
    if (stage_env && *stage_env)
        g_cuda.stage_bytes = (size_t)strtoull(stage_env, NULL, 10) << 20;
    if (g_cuda.stage_bytes < ((size_t)16 << 20)) g_cuda.stage_bytes = (size_t)16 << 20;
    g_cuda.q3_group = 256;
    const char *qgroup_env = getenv("K3_CUDA_Q3_GROUP");
    if (qgroup_env && *qgroup_env) g_cuda.q3_group = atoi(qgroup_env);
    if (g_cuda.q3_group < 8 || (g_cuda.q3_group & 7)) {
        fprintf(stderr, "k3_cuda: K3_CUDA_Q3_GROUP must be a positive multiple of 8\n");
        return -1;
    }

    double cache_gb = 18.0;
    const char *cache_env = getenv("K3_CUDA_CACHE_GB");
    if (cache_env && *cache_env) cache_gb = strtod(cache_env, NULL);
    if (cache_gb < 0.0) cache_gb = 0.0;
    size_t free_bytes = 0, total_bytes = 0;
    if (driver_error(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo") != 0)
        return -1;
    size_t requested = (size_t)(cache_gb * 1e9);
    double host_cache_gb = 40.0;
    const char *host_cache_env = getenv("K3_CUDA_HOST_CACHE_GB");
    if (host_cache_env && *host_cache_env) host_cache_gb = strtod(host_cache_env, NULL);
    if (host_cache_gb < 0.0) host_cache_gb = 0.0;
    g_cuda.host_cache_budget = (size_t)(host_cache_gb * 1e9);
    /* What must stay OUT of the cache: the quantisation staging buffer (chunked, so
     * K3_CUDA_STAGE_MB rather than the whole 2.35 GB lm_head), one expert-sized MXFP4
     * staging pair, the activation vectors and allocator slack. Erring tight here is
     * what pushes a trunk that would have fitted into the per-token PCIe tier, so it
     * is computed rather than guessed. */
    const size_t reserve_bytes = g_cuda.cpu_lane
        ? ((size_t)768 << 20)                       /* nothing big is ever staged */
        : (g_cuda.stage_bytes + ((size_t)768 << 20));
    size_t safe = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
    g_cuda.cache_budget = requested < safe ? requested : safe;
    g_cuda.cacheable = 1;
    g_cuda.enabled = 1;
    if (g_cuda.cache_budget) {
        /* Claim the whole cache up front. If the card cannot give it, halve and retry
         * rather than discovering it 90 layers into a warm-up. */
        while (g_cuda.cache_budget > ((size_t)1 << 30) &&
               arena_reserve(g_cuda.cache_budget) != 0)
            g_cuda.cache_budget = g_cuda.cache_budget / 4 * 3;
        if (!g_cuda.arena) {
            fprintf(stderr, "k3_cuda: could not reserve a packed-cache arena\n");
            return -1;
        }
    }
    fprintf(stderr,
            "k3_cuda: device %d %s, offloading matrices >= %.2f MB, "
            "persistent %s cache %.2f GB%s (%.2f/%.2f GB VRAM free/total)\n",
            device_index, g_cuda.name, (double)g_cuda.min_bytes / 1e6,
            g_cuda.q3_cache ? "Q3" : (g_cuda.q4_cache ? "Q4"
                                     : (g_cuda.mix ? "mixed Q4/Q3" : "BF16")),
            (double)g_cuda.cache_budget / 1e9,
            (g_cuda.q3_cache || g_cuda.q4_cache || g_cuda.mix)
                ? (g_cuda.q3_group == 128 ? ", group 128" :
                   g_cuda.q3_group == 256 ? ", group 256" : ", custom group")
                : "",
            (double)free_bytes / 1e9, (double)total_bytes / 1e9);
    return 0;
}

extern "C" int k3_cuda_enabled(void)
{
    return g_cuda.enabled;
}

extern "C" int k3_cuda_compressed_cache(void)
{
    return g_cuda.q3_cache || g_cuda.q4_cache || g_cuda.mix;
}

extern "C" int k3_cuda_compressed_complete(void)
{
    return k3_cuda_compressed_cache() && g_cuda.compressed_fallbacks == 0;
}

extern "C" void k3_cuda_set_cache_scope(int scope, int cacheable)
{
    g_cuda.cache_scope = scope;
    g_cuda.cacheable = cacheable != 0;
}

extern "C" int k3_cuda_matmul_bf16(float *y, const float *x, const uint16_t *weights,
                                    int in, int rows)
{
    const size_t wbytes = (size_t)in * (size_t)rows * sizeof(uint16_t);
    if (!g_cuda.enabled || wbytes < g_cuda.min_bytes) return 1;
    const size_t xbytes = (size_t)in * sizeof(float);
    const size_t ybytes = (size_t)rows * sizeof(float);
    if (reserve(&g_cuda.x, &g_cuda.x_cap, xbytes) != 0 ||
        reserve(&g_cuda.y, &g_cuda.y_cap, ybytes) != 0)
        return -1;

    const double t0 = now_s();

    K3CudaCacheEntry *e = NULL;
    int quantize = 0;
    const int r = cached_quant(weights, wbytes, in, rows, &e, &quantize);
    if (r < 0) return -1;
    if (r == 0) {
        if (quantize && quantize_entry(e, weights) != 0) return -1;

        CUdeviceptr packed = e->device, scales = e->scales;
        if (e->on_host) {
            /* The overflow tier: every token, every layer, over PCIe. Keeping this
             * branch unused is the entire reason the mixed format exists. */
            if (reserve(&g_cuda.qweights, &g_cuda.qweights_cap, e->packed_bytes) != 0 ||
                reserve(&g_cuda.qscales, &g_cuda.qscales_cap, e->scale_bytes) != 0)
                return -1;
            packed = g_cuda.qweights;
            scales = g_cuda.qscales;
            if (!quantize) {
                if (driver_error(cuMemcpyHtoDAsync(packed, e->host_packed,
                                                   e->packed_bytes, g_cuda.stream),
                                 "overflow packed H2D") != 0 ||
                    driver_error(cuMemcpyHtoDAsync(scales, e->host_scales,
                                                   e->scale_bytes, g_cuda.stream),
                                 "overflow scales H2D") != 0)
                    return -1;
                g_cuda.h2d_bytes += e->packed_bytes + e->scale_bytes;
            }
        }

        int qgroup = g_cuda.q3_group;
        if (driver_error(cuMemcpyHtoDAsync(g_cuda.x, x, xbytes, g_cuda.stream),
                         "input H2D") != 0)
            return -1;
        void *args[] = { &g_cuda.y, &g_cuda.x, &packed, &scales, &in, &rows, &qgroup };
        if (driver_error(cuLaunchKernel(e->bits == 3 ? g_cuda.q3_fn : g_cuda.q4_fn,
                                        (unsigned int)rows, 1, 1, 256, 1, 1,
                                        0, g_cuda.stream, args, NULL),
                         "packed gemv launch") != 0 ||
            driver_error(cuMemcpyDtoHAsync(y, g_cuda.y, ybytes, g_cuda.stream),
                         "output D2H") != 0 ||
            driver_error(cuStreamSynchronize(g_cuda.stream), "gemv synchronize") != 0)
            return -1;

        g_cuda.bf16_calls++;
        g_cuda.h2d_bytes += xbytes;
        g_cuda.d2h_bytes += ybytes;
        const double dt = now_s() - t0;
        g_cuda.wall_seconds += dt;
        if (e->on_host) {
            g_cuda.st_host_s += dt;
            g_cuda.st_host_n++;
            g_cuda.st_host_bytes += e->packed_bytes + e->scale_bytes;
        } else {
            g_cuda.st_dev_s += dt;
            g_cuda.st_dev_n++;
        }
        return 0;
    }

    /* THE CPU LANE. A matrix that did not win a VRAM slot has to be multiplied
     * somewhere, and the choice is not obvious: shipping it to the GPU costs a PCIe
     * crossing at a measured 18.3 GB/s, while multiplying it in place costs a sweep of
     * host RAM at roughly 50 GB/s on this machine. For a batch-1 GEMV, which is purely
     * memory bound and does no arithmetic worth moving data for, the CPU wins by more
     * than a factor of two -- and the exact preset has 87 GB per token that cannot fit
     * in VRAM at any setting. So: return 1 and let k3_mmw run the AVX2 kernel from the
     * bytes that are already in RAM. K3_CUDA_CPU_LANE=0 restores the upload path. */
    if (g_cuda.cpu_lane) return 1;

    /* Uncacheable: a BF16 matrix uploaded for this call only. Correct, and slow enough
     * that the run summary counts every one of them. */
    CUdeviceptr device_weights = 0;
    int upload_weights = 0;
    const int cache_result = cached_bf16(weights, wbytes, &device_weights,
                                         &upload_weights);
    if (cache_result < 0) return -1;
    if (cache_result > 0) {
        if (reserve(&g_cuda.weights, &g_cuda.weights_cap, wbytes) != 0) return -1;
        device_weights = g_cuda.weights;
        upload_weights = 1;
    }
    if ((upload_weights &&
         driver_error(cuMemcpyHtoDAsync(device_weights, weights, wbytes, g_cuda.stream),
                      "BF16 weights H2D") != 0) ||
        driver_error(cuMemcpyHtoDAsync(g_cuda.x, x, xbytes, g_cuda.stream),
                     "BF16 input H2D") != 0)
        return -1;
    void *args[] = { &g_cuda.y, &g_cuda.x, &device_weights, &in, &rows };
    if (driver_error(cuLaunchKernel(g_cuda.bf16_fn,
                                    (unsigned int)rows, 1, 1, 256, 1, 1,
                                    0, g_cuda.stream, args, NULL),
                     "bf16_gemv launch") != 0 ||
        driver_error(cuMemcpyDtoHAsync(y, g_cuda.y, ybytes, g_cuda.stream),
                     "BF16 output D2H") != 0 ||
        driver_error(cuStreamSynchronize(g_cuda.stream), "BF16 synchronize") != 0)
        return -1;

    g_cuda.bf16_calls++;
    g_cuda.h2d_bytes += (upload_weights ? wbytes : 0) + xbytes;
    g_cuda.d2h_bytes += ybytes;
    const double dt = now_s() - t0;
    g_cuda.wall_seconds += dt;
    if (upload_weights) {
        g_cuda.st_bf16_s += dt;
        g_cuda.st_bf16_n++;
        g_cuda.st_bf16_bytes += wbytes;
    } else {
        g_cuda.st_dev_s += dt;
        g_cuda.st_dev_n++;
    }
    return 0;
}

/* Router gate GEMV. The gate is 896 x 7168 per layer and the CPU router reads all
 * 25.7 MB of it, per layer, per token, with a double accumulator: 2.4 GB of RAM
 * traffic and 590M double MACs on the critical path of every step. Packed once into
 * the resident cache it costs 0.33 GB of VRAM for the whole model and the scores come
 * back in microseconds. The caller still applies the sigmoid and the top-k, which is
 * where the selection semantics live. */
extern "C" int k3_cuda_matmul_f32(float *y, const float *x, const float *weights,
                                  int in, int rows)
{
    const size_t wbytes = (size_t)in * (size_t)rows * sizeof(float);
    if (!g_cuda.enabled || !(g_cuda.mix || g_cuda.q3_cache || g_cuda.q4_cache))
        return 1;
    if ((in & 7) || !g_cuda.arena) return 1;
    const size_t xbytes = (size_t)in * sizeof(float);
    const size_t ybytes = (size_t)rows * sizeof(float);
    if (reserve(&g_cuda.x, &g_cuda.x_cap, xbytes) != 0 ||
        reserve(&g_cuda.y, &g_cuda.y_cap, ybytes) != 0)
        return -1;

    const double t0 = now_s();
    K3CudaCacheEntry *e = cache_find(weights, wbytes);
    if (!e) {
        const size_t sbytes = scale_size(in, rows);
        const size_t pbytes = packed_size(4, in, rows);
        if (!g_cuda.cacheable ||
            g_cuda.arena_used + pbytes + sbytes > g_cuda.arena_cap) return 1;
        CUdeviceptr packed = 0, scales = 0;
        if (arena_take(pbytes, &packed) != 0 || arena_take(sbytes, &scales) != 0)
            return 1;
        e = cache_push();
        if (!e) return -1;
        e->host = weights;
        e->bytes = wbytes;
        e->scope = g_cuda.cache_scope;
        e->bits = 4;
        e->src_f32 = 1;
        e->in = in;
        e->rows = rows;
        e->device = packed;
        e->scales = scales;
        e->packed_bytes = pbytes;
        e->scale_bytes = sbytes;
        g_cuda.cache_bytes = g_cuda.arena_used;
        g_cuda.cache_misses++;
        if (quantize_entry(e, weights) != 0) return -1;
    } else {
        g_cuda.cache_hits++;
    }

    int qgroup = g_cuda.q3_group;
    if (driver_error(cuMemcpyHtoDAsync(g_cuda.x, x, xbytes, g_cuda.stream),
                     "router input H2D") != 0)
        return -1;
    void *args[] = { &g_cuda.y, &g_cuda.x, &e->device, &e->scales, &in, &rows,
                     &qgroup };
    if (driver_error(cuLaunchKernel(g_cuda.q4_fn, (unsigned int)rows, 1, 1, 256, 1, 1,
                                    0, g_cuda.stream, args, NULL),
                     "router gemv launch") != 0 ||
        driver_error(cuMemcpyDtoHAsync(y, g_cuda.y, ybytes, g_cuda.stream),
                     "router output D2H") != 0 ||
        driver_error(cuStreamSynchronize(g_cuda.stream), "router synchronize") != 0)
        return -1;
    g_cuda.bf16_calls++;
    const double dt = now_s() - t0;
    g_cuda.wall_seconds += dt;
    g_cuda.st_dev_s += dt;
    g_cuda.st_dev_n++;
    return 0;
}

/* Page-lock a host buffer the engine already owns -- the expert arena. Experts are
 * copied to the GPU straight out of it, and a pageable copy runs at about 8 GB/s
 * because the driver stages it through its own pinned bounce buffer; pinned, the same
 * bytes move at PCIe speed. Failure is not fatal: it just stays pageable. */
extern "C" int k3_cuda_register_host(void *ptr, size_t bytes)
{
    if (!g_cuda.enabled || !ptr || !bytes) return 1;
    const CUresult r = cuMemHostRegister(ptr, bytes, 0);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "k3_cuda: expert arena stays pageable (%s)\n",
                name ? name : "?");
        return 1;
    }
    fprintf(stderr, "k3_cuda: pinned %.2f GB expert arena for PCIe-speed uploads\n",
            (double)bytes / 1e9);
    return 0;
}

extern "C" int k3_cuda_matmul_mxfp4(float *y, const float *x,
                                     const unsigned char *packed,
                                     const unsigned char *scales,
                                     int in, int rows, int group)
{
    const size_t pbytes = (size_t)rows * (size_t)(in / 2);
    const size_t sbytes = (size_t)rows * (size_t)((in + group - 1) / group);
    if (!g_cuda.enabled || pbytes + sbytes < g_cuda.min_bytes) return 1;
    const size_t xbytes = (size_t)in * sizeof(float);
    const size_t ybytes = (size_t)rows * sizeof(float);
    if (reserve(&g_cuda.weights, &g_cuda.weights_cap, pbytes) != 0 ||
        reserve(&g_cuda.scales, &g_cuda.scales_cap, sbytes) != 0 ||
        reserve(&g_cuda.x, &g_cuda.x_cap, xbytes) != 0 ||
        reserve(&g_cuda.y, &g_cuda.y_cap, ybytes) != 0)
        return -1;

    const double t0 = now_s();
    if (driver_error(cuMemcpyHtoDAsync(g_cuda.weights, packed, pbytes, g_cuda.stream),
                     "MXFP4 weights H2D") != 0 ||
        driver_error(cuMemcpyHtoDAsync(g_cuda.scales, scales, sbytes, g_cuda.stream),
                     "MXFP4 scales H2D") != 0 ||
        driver_error(cuMemcpyHtoDAsync(g_cuda.x, x, xbytes, g_cuda.stream),
                     "MXFP4 input H2D") != 0)
        return -1;
    void *args[] = {
        &g_cuda.y, &g_cuda.x, &g_cuda.weights, &g_cuda.scales, &in, &rows, &group
    };
    if (driver_error(cuLaunchKernel(g_cuda.mxfp4_fn,
                                    (unsigned int)rows, 1, 1, 256, 1, 1,
                                    0, g_cuda.stream, args, NULL),
                     "mxfp4_gemv launch") != 0 ||
        driver_error(cuMemcpyDtoHAsync(y, g_cuda.y, ybytes, g_cuda.stream),
                     "MXFP4 output D2H") != 0 ||
        driver_error(cuStreamSynchronize(g_cuda.stream), "MXFP4 synchronize") != 0)
        return -1;

    g_cuda.mxfp4_calls++;
    g_cuda.h2d_bytes += pbytes + sbytes + xbytes;
    g_cuda.d2h_bytes += ybytes;
    const double dt = now_s() - t0;
    g_cuda.wall_seconds += dt;
    g_cuda.st_mxfp4_s += dt;
    g_cuda.st_mxfp4_n++;
    g_cuda.st_mxfp4_bytes += pbytes + sbytes;
    return 0;
}

/* Per-step attribution, printed by the decode loop under K3_CUDA_PROFILE. */
extern "C" void k3_cuda_step_reset(void)
{
    g_cuda.st_dev_s = g_cuda.st_host_s = g_cuda.st_bf16_s = g_cuda.st_mxfp4_s = 0.0;
    g_cuda.st_dev_n = g_cuda.st_host_n = g_cuda.st_bf16_n = g_cuda.st_mxfp4_n = 0;
    g_cuda.st_host_bytes = g_cuda.st_bf16_bytes = g_cuda.st_mxfp4_bytes = 0;
}

extern "C" void k3_cuda_step_report(double step_seconds)
{
    const double backend = g_cuda.st_dev_s + g_cuda.st_host_s +
                           g_cuda.st_bf16_s + g_cuda.st_mxfp4_s;
    fprintf(stderr,
            "  cuda step: resident %llu calls %.3fs | overflow %llu calls %.2f GB "
            "%.3fs | bf16-miss %llu calls %.2f GB %.3fs | expert %llu calls %.2f GB "
            "%.3fs | backend %.3fs of %.3fs (cpu %.3fs)\n",
            g_cuda.st_dev_n, g_cuda.st_dev_s,
            g_cuda.st_host_n, (double)g_cuda.st_host_bytes / 1e9, g_cuda.st_host_s,
            g_cuda.st_bf16_n, (double)g_cuda.st_bf16_bytes / 1e9, g_cuda.st_bf16_s,
            g_cuda.st_mxfp4_n, (double)g_cuda.st_mxfp4_bytes / 1e9, g_cuda.st_mxfp4_s,
            backend, step_seconds, step_seconds - backend);
}

extern "C" void k3_cuda_report(void)
{
    const unsigned long long calls = g_cuda.bf16_calls + g_cuda.mxfp4_calls;
    const double gb = (double)g_cuda.h2d_bytes / 1e9;
    fprintf(stderr,
            "k3_cuda: %llu calls (%llu BF16, %llu MXFP4), %.2f GB H2D, %.2f GB D2H, "
            "%.2f s backend wall, %.2f GB/s effective H2D\n",
            calls, g_cuda.bf16_calls, g_cuda.mxfp4_calls, gb,
            (double)g_cuda.d2h_bytes / 1e9, g_cuda.wall_seconds,
            g_cuda.wall_seconds > 0.0 ? gb / g_cuda.wall_seconds : 0.0);
    fprintf(stderr,
            "k3_cuda: persistent %s cache %.2f/%.2f GB, %llu hits, %llu misses, "
            "%llu Q4->Q3 downgrades, %llu uncached\n",
            g_cuda.q3_cache ? "Q3" : (g_cuda.q4_cache ? "Q4"
                                     : (g_cuda.mix ? "mixed Q4/Q3" : "BF16")),
            (double)g_cuda.cache_bytes / 1e9,
            (double)g_cuda.cache_budget / 1e9,
            g_cuda.cache_hits, g_cuda.cache_misses,
            g_cuda.downgrades, g_cuda.compressed_fallbacks);
    if (g_cuda.host_cache_bytes)
        fprintf(stderr,
                "k3_cuda: compressed host overflow %.2f/%.2f GB (%.2f GB pinned)\n",
                (double)g_cuda.host_cache_bytes / 1e9,
                (double)g_cuda.host_cache_budget / 1e9,
                (double)g_cuda.host_pinned_bytes / 1e9);
}

extern "C" void k3_cuda_shutdown(void)
{
    if (!g_cuda.enabled) return;
    cuStreamSynchronize(g_cuda.stream);
    for (size_t i = 0; i < g_cuda.cache_len; i++) {
        /* Packed entries live inside the arena and are freed with it; only the plain
         * BF16 tier owns its device allocation. */
        if (g_cuda.cache[i].bits == 0 && g_cuda.cache[i].device)
            cuMemFree(g_cuda.cache[i].device);
        if (g_cuda.cache[i].host_pinned) {
            if (g_cuda.cache[i].host_packed) cuMemFreeHost(g_cuda.cache[i].host_packed);
            if (g_cuda.cache[i].host_scales) cuMemFreeHost(g_cuda.cache[i].host_scales);
        } else {
            free(g_cuda.cache[i].host_packed);
            free(g_cuda.cache[i].host_scales);
        }
    }
    free(g_cuda.cache);
    if (g_cuda.arena) cuMemFree(g_cuda.arena);
    if (g_cuda.weights) cuMemFree(g_cuda.weights);
    if (g_cuda.scales) cuMemFree(g_cuda.scales);
    if (g_cuda.x) cuMemFree(g_cuda.x);
    if (g_cuda.y) cuMemFree(g_cuda.y);
    if (g_cuda.qweights) cuMemFree(g_cuda.qweights);
    if (g_cuda.qscales) cuMemFree(g_cuda.qscales);
    if (g_cuda.module) cuModuleUnload(g_cuda.module);
    if (g_cuda.stream) cuStreamDestroy(g_cuda.stream);
    if (g_cuda.context) cuCtxDestroy(g_cuda.context);
    memset(&g_cuda, 0, sizeof g_cuda);
}
