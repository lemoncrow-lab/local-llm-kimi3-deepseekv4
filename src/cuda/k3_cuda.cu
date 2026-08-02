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

__global__ void block_scale(const unsigned short *weights, float *scales,
                            int in, int rows, int group, int qmax)
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
    float m = 0.0f;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float v = __uint_as_float((unsigned int)w[i] << 16);
        m = fmaxf(m, fabsf(v));
    }
    __shared__ float wm[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    m = warp_max(m);
    if (lane == 0) wm[warp] = m;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        m = lane < nw ? wm[lane] : 0.0f;
        m = warp_max(m);
        if (lane == 0) scales[si] = m > 0.0f ? m / (float)qmax : 1.0f;
    }
}

__global__ void q3_pack(const unsigned short *weights, const float *scales,
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
        const float inv = 1.0f / scales[row * ngrp + (g * 8) / group];
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
                        const unsigned char *packed, const float *scales,
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
        const float scale = scales[row * ngrp + (g * 8) / group];
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

__global__ void q4_pack(const unsigned short *weights, const float *scales,
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
        const float inv = 1.0f / scales[row * ngrp + (g * 2) / group];
        int a = (int)rintf(__uint_as_float((unsigned int)w[g * 2] << 16) * inv);
        int b = (int)rintf(__uint_as_float((unsigned int)w[g * 2 + 1] << 16) * inv);
        a = a < -7 ? -7 : (a > 7 ? 7 : a);
        b = b < -7 ? -7 : (b > 7 ? 7 : b);
        p[g] = (unsigned char)((a + 7) | ((b + 7) << 4));
    }
}

__global__ void q4_gemv(float *out, const float *x,
                        const unsigned char *packed, const float *scales,
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
        const float scale = scales[row * ngrp + (g * 2) / group];
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
    int q3;
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

/* Return a persistent device copy for a stable host allocation.  A streamed trunk
 * reuses host addresses for different layers, so the caller explicitly marks only
 * pinned/resident tensors as cacheable.  Once the budget is full we keep the existing
 * entries and use the scratch allocation; eviction would merely recreate a cyclic-scan
 * pathology on the GPU. */
static int cached_bf16(const void *host, size_t bytes, CUdeviceptr *device,
                       int *needs_upload)
{
    *needs_upload = 0;
    if (!g_cuda.cacheable || g_cuda.cache_budget == 0 ||
        g_cuda.q3_cache || g_cuda.q4_cache) return 1;
    for (size_t i = 0; i < g_cuda.cache_len; i++) {
        K3CudaCacheEntry *e = &g_cuda.cache[i];
        if (!e->q3 && e->scope == g_cuda.cache_scope &&
            e->host == host && e->bytes == bytes) {
            *device = e->device;
            g_cuda.cache_hits++;
            return 0;
        }
    }

    g_cuda.cache_misses++;
    if (bytes > g_cuda.cache_budget - g_cuda.cache_bytes) return 1;
    if (g_cuda.cache_len == g_cuda.cache_cap) {
        size_t cap = g_cuda.cache_cap ? g_cuda.cache_cap * 2 : 128;
        void *p = realloc(g_cuda.cache, cap * sizeof(*g_cuda.cache));
        if (!p) return -1;
        g_cuda.cache = (K3CudaCacheEntry *)p;
        g_cuda.cache_cap = cap;
    }

    CUdeviceptr ptr = 0;
    if (driver_error(cuMemAlloc(&ptr, bytes), "cuMemAlloc(cache)") != 0) return -1;
    K3CudaCacheEntry *e = &g_cuda.cache[g_cuda.cache_len++];
    e->host = host;
    e->bytes = bytes;
    e->scope = g_cuda.cache_scope;
    e->q3 = 0;
    e->device = ptr;
    e->scales = 0;
    e->host_packed = e->host_scales = NULL;
    e->packed_bytes = bytes;
    e->scale_bytes = 0;
    e->host_pinned = 0;
    g_cuda.cache_bytes += bytes;
    *device = ptr;
    *needs_upload = 1;
    return 0;
}

static int cached_q3(const void *host, size_t source_bytes, int in, int rows,
                     K3CudaCacheEntry **entry, int *needs_quantize)
{
    *needs_quantize = 0;
    if (!g_cuda.cacheable || !g_cuda.q3_cache || (in & 7)) return 1;
    for (size_t i = 0; i < g_cuda.cache_len; i++) {
        K3CudaCacheEntry *e = &g_cuda.cache[i];
        if (e->q3 && e->scope == g_cuda.cache_scope &&
            e->host == host && e->bytes == source_bytes) {
            *entry = e;
            g_cuda.cache_hits++;
            return 0;
        }
    }

    g_cuda.cache_misses++;
    const size_t packed_bytes = (size_t)rows * (size_t)(in / 8) * 3;
    const int group = g_cuda.q3_group;
    const size_t scale_bytes =
        (size_t)rows * (size_t)((in + group - 1) / group) * sizeof(float);
    const size_t storage = packed_bytes + scale_bytes;
    if (storage > g_cuda.cache_budget - g_cuda.cache_bytes) return 1;
    if (g_cuda.cache_len == g_cuda.cache_cap) {
        size_t cap = g_cuda.cache_cap ? g_cuda.cache_cap * 2 : 128;
        void *p = realloc(g_cuda.cache, cap * sizeof(*g_cuda.cache));
        if (!p) return -1;
        g_cuda.cache = (K3CudaCacheEntry *)p;
        g_cuda.cache_cap = cap;
    }

    CUdeviceptr packed = 0, scales = 0;
    if (driver_error(cuMemAlloc(&packed, packed_bytes), "cuMemAlloc(q3)") != 0 ||
        driver_error(cuMemAlloc(&scales, scale_bytes), "cuMemAlloc(q3 scales)") != 0) {
        if (packed) cuMemFree(packed);
        if (scales) cuMemFree(scales);
        return -1;
    }
    K3CudaCacheEntry *e = &g_cuda.cache[g_cuda.cache_len++];
    e->host = host;
    e->bytes = source_bytes;
    e->scope = g_cuda.cache_scope;
    e->q3 = 1;
    e->device = packed;
    e->scales = scales;
    e->host_packed = e->host_scales = NULL;
    e->packed_bytes = packed_bytes;
    e->scale_bytes = scale_bytes;
    e->host_pinned = 0;
    g_cuda.cache_bytes += storage;
    *entry = e;
    *needs_quantize = 1;
    return 0;
}

static int cached_q4(const void *host, size_t source_bytes, int in, int rows,
                     K3CudaCacheEntry **entry, int *needs_quantize)
{
    *needs_quantize = 0;
    if (!g_cuda.cacheable || !g_cuda.q4_cache || (in & 1)) {
        if (g_cuda.q4_cache) g_cuda.compressed_fallbacks++;
        return 1;
    }
    for (size_t i = 0; i < g_cuda.cache_len; i++) {
        K3CudaCacheEntry *e = &g_cuda.cache[i];
        if ((e->q3 == 2 || e->q3 == 3) && e->scope == g_cuda.cache_scope &&
            e->host == host && e->bytes == source_bytes) {
            *entry = e;
            g_cuda.cache_hits++;
            return 0;
        }
    }

    g_cuda.cache_misses++;
    const int group = g_cuda.q3_group;
    const size_t packed_bytes = (size_t)rows * (size_t)(in / 2);
    const size_t scale_bytes =
        (size_t)rows * (size_t)((in + group - 1) / group) * sizeof(float);
    const size_t storage = packed_bytes + scale_bytes;
    const int on_device = storage <= g_cuda.cache_budget - g_cuda.cache_bytes;
    if (!on_device && storage > g_cuda.host_cache_budget - g_cuda.host_cache_bytes) {
        g_cuda.compressed_fallbacks++;
        return 1;
    }
    if (g_cuda.cache_len == g_cuda.cache_cap) {
        size_t cap = g_cuda.cache_cap ? g_cuda.cache_cap * 2 : 128;
        void *p = realloc(g_cuda.cache, cap * sizeof(*g_cuda.cache));
        if (!p) return -1;
        g_cuda.cache = (K3CudaCacheEntry *)p;
        g_cuda.cache_cap = cap;
    }

    CUdeviceptr packed = 0, scales = 0;
    void *host_packed = NULL, *host_scales = NULL;
    if (on_device) {
        if (driver_error(cuMemAlloc(&packed, packed_bytes), "cuMemAlloc(q4)") != 0 ||
            driver_error(cuMemAlloc(&scales, scale_bytes), "cuMemAlloc(q4 scales)") != 0) {
            if (packed) cuMemFree(packed);
            if (scales) cuMemFree(scales);
            return -1;
        }
    } else {
        CUresult hp = cuMemHostAlloc(&host_packed, packed_bytes, CU_MEMHOSTALLOC_PORTABLE);
        CUresult hs = hp == CUDA_SUCCESS
            ? cuMemHostAlloc(&host_scales, scale_bytes, CU_MEMHOSTALLOC_PORTABLE)
            : CUDA_ERROR_OUT_OF_MEMORY;
        if (hp != CUDA_SUCCESS || hs != CUDA_SUCCESS) {
            if (hp == CUDA_SUCCESS) cuMemFreeHost(host_packed);
            if (hs == CUDA_SUCCESS) cuMemFreeHost(host_scales);
            host_packed = malloc(packed_bytes);
            host_scales = malloc(scale_bytes);
            if (!host_packed || !host_scales) {
                free(host_packed);
                free(host_scales);
                return -1;
            }
        }
    }
    K3CudaCacheEntry *e = &g_cuda.cache[g_cuda.cache_len++];
    e->host = host;
    e->bytes = source_bytes;
    e->scope = g_cuda.cache_scope;
    e->q3 = on_device ? 2 : 3;
    e->device = packed;
    e->scales = scales;
    e->host_packed = host_packed;
    e->host_scales = host_scales;
    e->packed_bytes = packed_bytes;
    e->scale_bytes = scale_bytes;
    e->host_pinned = !on_device && host_packed &&
                     cuMemHostGetDevicePointer(&packed, host_packed, 0) == CUDA_SUCCESS;
    if (on_device) {
        g_cuda.cache_bytes += storage;
    } else {
        g_cuda.host_cache_bytes += storage;
        if (e->host_pinned) g_cuda.host_pinned_bytes += storage;
    }
    *entry = e;
    *needs_quantize = 1;
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
                     "cuModuleGetFunction(q4_gemv)") != 0;
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
        driver_error(cuCtxCreate(&g_cuda.context, 0, device), "cuCtxCreate") != 0 ||
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
    if (format && *format && !g_cuda.q3_cache && !g_cuda.q4_cache &&
        strcmp(format, "bf16")) {
        fprintf(stderr, "k3_cuda: K3_CUDA_CACHE_FORMAT must be bf16, q3 or q4\n");
        return -1;
    }
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
    const size_t reserve_bytes = (g_cuda.q3_cache || g_cuda.q4_cache)
        ? ((size_t)5 << 29)  /* 2.5 GiB: largest BF16 staging tensor is lm_head */
        : ((size_t)4 << 30);
    size_t safe = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
    g_cuda.cache_budget = requested < safe ? requested : safe;
    g_cuda.cacheable = 1;
    g_cuda.enabled = 1;
    fprintf(stderr,
            "k3_cuda: device %d %s, offloading matrices >= %.2f MB, "
            "persistent %s cache %.2f GB%s (%.2f/%.2f GB VRAM free/total)\n",
            device_index, g_cuda.name, (double)g_cuda.min_bytes / 1e6,
            g_cuda.q3_cache ? "Q3" : (g_cuda.q4_cache ? "Q4" : "BF16"),
            (double)g_cuda.cache_budget / 1e9,
            (g_cuda.q3_cache || g_cuda.q4_cache)
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
    return g_cuda.q3_cache || g_cuda.q4_cache;
}

extern "C" int k3_cuda_compressed_complete(void)
{
    return (g_cuda.q3_cache || g_cuda.q4_cache) && g_cuda.compressed_fallbacks == 0;
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

    K3CudaCacheEntry *q3 = NULL;
    int quantize = 0;
    int q3_result = cached_q3(weights, wbytes, in, rows, &q3, &quantize);
    if (q3_result < 0) return -1;
    if (q3_result == 0) {
        if (quantize) {
            if (reserve(&g_cuda.weights, &g_cuda.weights_cap, wbytes) != 0)
                return -1;
            if (driver_error(cuMemcpyHtoDAsync(g_cuda.weights, weights, wbytes,
                                               g_cuda.stream),
                             "Q3 source weights H2D") != 0)
                return -1;
            int qgroup = g_cuda.q3_group;
            const unsigned int scale_blocks =
                (unsigned int)((size_t)rows * (size_t)((in + qgroup - 1) / qgroup));
            int qmax = 3;
            void *scale_args[] = {
                &g_cuda.weights, &q3->scales, &in, &rows, &qgroup, &qmax
            };
            if (driver_error(cuLaunchKernel(g_cuda.block_scale_fn,
                                            scale_blocks, 1, 1, 256, 1, 1,
                                            0, g_cuda.stream, scale_args, NULL),
                             "q3_scale launch") != 0)
                return -1;
            void *pack_args[] = {
                &g_cuda.weights, &q3->scales, &q3->device, &in, &rows, &qgroup
            };
            if (driver_error(cuLaunchKernel(g_cuda.q3_pack_fn,
                                            (unsigned int)rows, 1, 1, 256, 1, 1,
                                            0, g_cuda.stream, pack_args, NULL),
                             "q3_pack launch") != 0)
                return -1;
        }
        if (driver_error(cuMemcpyHtoDAsync(g_cuda.x, x, xbytes, g_cuda.stream),
                         "Q3 input H2D") != 0)
            return -1;
        int qgroup = g_cuda.q3_group;
        void *q3_args[] = {
            &g_cuda.y, &g_cuda.x, &q3->device, &q3->scales, &in, &rows, &qgroup
        };
        if (driver_error(cuLaunchKernel(g_cuda.q3_fn,
                                        (unsigned int)rows, 1, 1, 256, 1, 1,
                                        0, g_cuda.stream, q3_args, NULL),
                         "q3_gemv launch") != 0 ||
            driver_error(cuMemcpyDtoHAsync(y, g_cuda.y, ybytes, g_cuda.stream),
                         "Q3 output D2H") != 0 ||
            driver_error(cuStreamSynchronize(g_cuda.stream), "Q3 synchronize") != 0)
            return -1;
        g_cuda.bf16_calls++;
        g_cuda.h2d_bytes += (quantize ? wbytes : 0) + xbytes;
        g_cuda.d2h_bytes += ybytes;
        g_cuda.wall_seconds += now_s() - t0;
        return 0;
    }

    K3CudaCacheEntry *q4 = NULL;
    int quantize_q4 = 0;
    int q4_result = cached_q4(weights, wbytes, in, rows, &q4, &quantize_q4);
    if (q4_result < 0) return -1;
    if (q4_result == 0) {
        int qgroup = g_cuda.q3_group;
        const int host_q4 = q4->q3 == 3;
        CUdeviceptr q4_weights = q4->device;
        CUdeviceptr q4_scales = q4->scales;
        if (host_q4) {
            if (reserve(&g_cuda.qweights, &g_cuda.qweights_cap, q4->packed_bytes) != 0 ||
                reserve(&g_cuda.qscales, &g_cuda.qscales_cap, q4->scale_bytes) != 0)
                return -1;
            q4_weights = g_cuda.qweights;
            q4_scales = g_cuda.qscales;
        }
        if (quantize_q4) {
            if (reserve(&g_cuda.weights, &g_cuda.weights_cap, wbytes) != 0)
                return -1;
            if (driver_error(cuMemcpyHtoDAsync(g_cuda.weights, weights, wbytes,
                                               g_cuda.stream),
                             "Q4 source weights H2D") != 0)
                return -1;
            const unsigned int scale_blocks =
                (unsigned int)((size_t)rows * (size_t)((in + qgroup - 1) / qgroup));
            int qmax = 7;
            void *scale_args[] = {
                &g_cuda.weights, &q4_scales, &in, &rows, &qgroup, &qmax
            };
            if (driver_error(cuLaunchKernel(g_cuda.block_scale_fn,
                                            scale_blocks, 1, 1, 256, 1, 1,
                                            0, g_cuda.stream, scale_args, NULL),
                             "Q4 block_scale launch") != 0)
                return -1;
            void *pack_args[] = {
                &g_cuda.weights, &q4_scales, &q4_weights, &in, &rows, &qgroup
            };
            if (driver_error(cuLaunchKernel(g_cuda.q4_pack_fn,
                                            (unsigned int)rows, 1, 1, 256, 1, 1,
                                            0, g_cuda.stream, pack_args, NULL),
                             "q4_pack launch") != 0)
                return -1;
            if (host_q4 &&
                (driver_error(cuMemcpyDtoHAsync(q4->host_packed, q4_weights,
                                                q4->packed_bytes, g_cuda.stream),
                              "Q4 packed D2H") != 0 ||
                 driver_error(cuMemcpyDtoHAsync(q4->host_scales, q4_scales,
                                                q4->scale_bytes, g_cuda.stream),
                              "Q4 scales D2H") != 0 ||
                 driver_error(cuStreamSynchronize(g_cuda.stream),
                              "Q4 host-cache synchronize") != 0))
                return -1;
        } else if (host_q4) {
            if (driver_error(cuMemcpyHtoDAsync(q4_weights, q4->host_packed,
                                               q4->packed_bytes, g_cuda.stream),
                             "Q4 packed H2D") != 0 ||
                driver_error(cuMemcpyHtoDAsync(q4_scales, q4->host_scales,
                                               q4->scale_bytes, g_cuda.stream),
                             "Q4 scales H2D") != 0)
                return -1;
        }
        if (driver_error(cuMemcpyHtoDAsync(g_cuda.x, x, xbytes, g_cuda.stream),
                         "Q4 input H2D") != 0)
            return -1;
        void *q4_args[] = {
            &g_cuda.y, &g_cuda.x, &q4_weights, &q4_scales, &in, &rows, &qgroup
        };
        if (driver_error(cuLaunchKernel(g_cuda.q4_fn,
                                        (unsigned int)rows, 1, 1, 256, 1, 1,
                                        0, g_cuda.stream, q4_args, NULL),
                         "q4_gemv launch") != 0 ||
            driver_error(cuMemcpyDtoHAsync(y, g_cuda.y, ybytes, g_cuda.stream),
                         "Q4 output D2H") != 0 ||
            driver_error(cuStreamSynchronize(g_cuda.stream), "Q4 synchronize") != 0)
            return -1;
        g_cuda.bf16_calls++;
        g_cuda.h2d_bytes += (quantize_q4 ? wbytes : 0)
                            + (host_q4 && !quantize_q4
                               ? q4->packed_bytes + q4->scale_bytes : 0)
                            + xbytes;
        g_cuda.d2h_bytes += ybytes
                          + (host_q4 && quantize_q4
                             ? q4->packed_bytes + q4->scale_bytes : 0);
        g_cuda.wall_seconds += now_s() - t0;
        return 0;
    }

    CUdeviceptr device_weights = 0;
    int upload_weights = 0;
    int cache_result = cached_bf16(weights, wbytes, &device_weights, &upload_weights);
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
    g_cuda.wall_seconds += now_s() - t0;
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
    g_cuda.wall_seconds += now_s() - t0;
    return 0;
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
            "k3_cuda: persistent %s cache %.2f/%.2f GB, %llu hits, %llu misses\n",
            g_cuda.q3_cache ? "Q3" : (g_cuda.q4_cache ? "Q4" : "BF16"),
            (double)g_cuda.cache_bytes / 1e9,
            (double)g_cuda.cache_budget / 1e9,
            g_cuda.cache_hits, g_cuda.cache_misses);
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
        if (g_cuda.cache[i].device) cuMemFree(g_cuda.cache[i].device);
        if (g_cuda.cache[i].scales) cuMemFree(g_cuda.cache[i].scales);
        if (g_cuda.cache[i].host_pinned) {
            if (g_cuda.cache[i].host_packed) cuMemFreeHost(g_cuda.cache[i].host_packed);
            if (g_cuda.cache[i].host_scales) cuMemFreeHost(g_cuda.cache[i].host_scales);
        } else {
            free(g_cuda.cache[i].host_packed);
            free(g_cuda.cache[i].host_scales);
        }
    }
    free(g_cuda.cache);
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
