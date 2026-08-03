"""Two-level expert cache + streaming loader for DeepSeek-V4-Flash's routed experts.

Measured on this checkpoint (47 generated tokens, 12 126 expert accesses): routing is
very skewed.  Only 28.4% of the 11 008 (layer, expert) pairs are ever touched, the
hottest 5% of those (1.94 GiB) serve 32% of accesses, and 37% of a token's experts were
also used by the token before it.  So most of the 3.21 GiB a token nominally needs need
not come off disk at all:

    LRU size    2 GiB    4     8     16    32    48
    hit rate     0%     36%   51%   64%   74%   74%

Hence two caches, both keyed on (layer, expert):
  * a VRAM pool -- a hit costs nothing, the weights are already where the kernel wants
    them.  The grouped fp4 GEMM gathers by slot index, so cached experts need no copy.
  * a pinned-host pool -- a hit is a straight DMA at 18.9 GB/s instead of a pread at
    9.2 GB/s (dm-crypt bound on this box), and misses pread *into* the pinned slot, so
    there is never an extra CPU copy.

Eviction needs no extra fence in the decode path: MoE.forward reads the routing indices
back to the host immediately before calling get_layer, which synchronises the stream.
release() covers prefill, where several chunks are issued per layer without a readback.
"""
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch

W1_B = 2048 * 2048      # w1.weight, fp4-packed bytes
S1_B = 2048 * 128       # w1.scale, e8m0
W13_B = 2 * W1_B        # w1 then w3, contiguous -> logical [4096, 4096] fp4
S13_B = 2 * S1_B
W2_B = 4096 * 1024
S2_B = 4096 * 64
EXPERT_B = W13_B + S13_B + W2_B + S2_B      # 12.75 MiB

# (offset within a host slot, checkpoint tensor suffix, bytes)
PARTS = (
    (0, "w1.weight", W1_B),
    (W1_B, "w3.weight", W1_B),
    (W13_B, "w1.scale", S1_B),
    (W13_B + S1_B, "w3.scale", S1_B),
    (W13_B + S13_B, "w2.weight", W2_B),
    (W13_B + S13_B + W2_B, "w2.scale", S2_B),
)
# (offset within a slot, bytes) per GPU pool, in pool order
REGIONS = ((0, W13_B), (W13_B, S13_B), (W13_B + S13_B, W2_B), (W13_B + S13_B + W2_B, S2_B))

# cudaHostAlloc on this box takes 64 GiB in one call and fails at 72, but happily pins
# 96 GiB across 8 GiB slabs -- so the host pool is slabbed, sized to whole experts.
SLAB_EXPERTS = (8 << 30) // EXPERT_B


class ExpertStream:
    # left free for cuBLAS/triton workspaces, prefill activations and fragmentation.
    # cublasCreate() fails with ALLOC_FAILED if the cache eats into this.
    VRAM_RESERVE = 2.5 * (1 << 30)

    def __init__(self, w, vram_gib=0.0, ram_gib=24.0, threads=32, split=1 << 20,
                 max_experts=64, device="cuda"):
        self.w = w
        self.device = device
        self.E = max_experts
        self.split = split
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="dsv4-io")
        self.copy_stream = torch.cuda.Stream(device=device)
        self.fence = torch.cuda.Event()
        self.fence.record()

        self.fds = {}
        for path, *_ in w.index.values():
            if path not in self.fds:
                self.fds[path] = os.open(path, os.O_RDONLY)

        if vram_gib <= 0:                    # auto: whatever is left after the trunk
            vram_b = torch.cuda.mem_get_info(device)[0] - self.VRAM_RESERVE
        else:
            vram_b = vram_gib * (1 << 30)
        self.Ng = max(max_experts * 2, int(vram_b) // EXPERT_B)
        self.Nh = max(max_experts * 2, int(ram_gib * (1 << 30)) // EXPERT_B)
        # one flat pool so a miss is a single DMA; the four logical views are strided
        # on the expert axis, which the grouped GEMM takes as an explicit stride.
        self.gflat = torch.empty((self.Ng, EXPERT_B), dtype=torch.uint8, device=device)
        self.gpu = tuple(
            torch.as_strided(self.gflat, (self.Ng, 4096, n // 4096),
                             (EXPERT_B, n // 4096, 1), off)
            for off, n in REGIONS)
        self.hper = SLAB_EXPERTS
        self.hslabs, self.hmvs, done = [], [], 0
        while done < self.Nh:
            n = min(self.hper, self.Nh - done)
            try:
                t = torch.empty(n * EXPERT_B, dtype=torch.uint8, pin_memory=True)
            except (torch.AcceleratorError, RuntimeError, MemoryError) as exc:
                if done < 2 * max_experts:
                    raise
                print(f"stream: pinned only {done * EXPERT_B / 2**30:.0f} GiB of the "
                      f"requested {ram_gib:.0f} ({type(exc).__name__})", flush=True)
                break
            self.hslabs.append(t)
            self.hmvs.append(memoryview(t.numpy()))
            done += n
        self.Nh = done

        self.gmap, self.hmap = OrderedDict(), OrderedDict()
        self.gfree = list(range(self.Ng))
        self.hfree = list(range(self.Nh))
        self.idx_host = torch.empty(max_experts, dtype=torch.int32, pin_memory=True)
        self.idx_np = self.idx_host.numpy()
        self.idx_dev = torch.empty(max_experts, dtype=torch.int32, device=device)

        self.bytes = self.fetches = 0
        self.hit_g = self.hit_h = self.miss = 0
        self.t_fetch = self.t_expert = 0.0

    # ------------------------------------------------------------------ slots
    def _alloc_g(self, key):
        if self.gfree:
            s = self.gfree.pop()
        else:
            _, s = self.gmap.popitem(last=False)
        self.gmap[key] = s
        return s

    def _alloc_h(self, key):
        if self.hfree:
            s = self.hfree.pop()
        else:
            _, s = self.hmap.popitem(last=False)
        self.hmap[key] = s
        return s

    # ------------------------------------------------------------------- i/o
    def _read(self, mv, path, off, n):
        fd, got = self.fds[path], 0
        while got < n:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                raise IOError(f"short read {got}/{n} at {off} in {path}")
            got += r

    def _submit(self, layer, e, hslot):
        """pread the six source tensors straight into the pinned slot."""
        stem = f"layers.{layer}.ffn.experts.{e}."
        mv = self.hmvs[hslot // self.hper]
        base = (hslot % self.hper) * EXPERT_B
        futs = []
        for off, name, n in PARTS:
            path, soff, snb, _, _ = self.w.index[stem + name]
            assert snb == n, (stem + name, snb, n)
            # split big reads: pread on this box scales 3.1 -> 9.2 GB/s from queue
            # depth 4 to 32, so one 4 MiB read per thread leaves the device idle.
            for p in range(0, n, self.split):
                m = min(self.split, n - p)
                a = base + off + p
                futs.append(self.pool.submit(self._read, mv[a:a + m], path, soff + p, m))
        return futs

    def _h2d(self, gslot, hslot):
        slab = self.hslabs[hslot // self.hper]
        base = (hslot % self.hper) * EXPERT_B
        self.gflat[gslot].copy_(slab[base:base + EXPERT_B], non_blocking=True)

    # ----------------------------------------------------------------- public
    def get_layer(self, layer, experts):
        """Make `experts` of `layer` resident; returns (slot index tensor, pools)."""
        t0 = time.perf_counter()
        E = len(experts)
        assert E <= self.E, f"{E} experts > buffer of {self.E}"
        self.fence.synchronize()            # kernels reading evictable slots have retired

        slots = []
        warm, cold = [], []                 # (gslot, hslot) needing DMA / disk + DMA
        for e in experts:
            k = (layer, e)
            g = self.gmap.get(k)
            if g is not None:
                self.gmap.move_to_end(k)
                self.hit_g += 1
            else:
                g = self._alloc_g(k)
                h = self.hmap.get(k)
                if h is not None:
                    self.hmap.move_to_end(k)
                    self.hit_h += 1
                    warm.append((g, h))
                else:
                    h = self._alloc_h(k)
                    self.miss += 1
                    cold.append((g, h, e))
            slots.append(g)

        jobs = [(g, h, self._submit(layer, e, h)) for g, h, e in cold]
        with torch.cuda.stream(self.copy_stream):
            for g, h in warm:               # already in RAM: start the DMA immediately
                self._h2d(g, h)
            for g, h, futs in jobs:
                for f in futs:
                    f.result()
                self._h2d(g, h)
        torch.cuda.current_stream().wait_stream(self.copy_stream)

        self.idx_np[:E] = slots
        idx = self.idx_dev[:E]
        idx.copy_(self.idx_host[:E], non_blocking=True)

        self.bytes += E * EXPERT_B
        self.fetches += E
        self.t_fetch += time.perf_counter() - t0
        return idx, self.gpu

    def release(self):
        self.fence.record()

    def stats(self):
        n = max(1, self.hit_g + self.hit_h + self.miss)
        return (f"cache {100*self.hit_g/n:.0f}% vram / {100*self.hit_h/n:.0f}% ram / "
                f"{100*self.miss/n:.0f}% disk")

    def close(self):
        self.pool.shutdown(wait=False)
        for fd in self.fds.values():
            os.close(fd)
