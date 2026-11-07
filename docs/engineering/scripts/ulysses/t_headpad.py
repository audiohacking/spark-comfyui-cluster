"""Ulysses at a degree that does not divide the head count, by zero head padding.

3 CPU ranks over gloo, yunchang's real SeqAllToAll4D. H3's layout: 56 heads, a
packed length that is not a multiple of 3 (sequence pad rows, excluded as keys
the way dgx-monarch's full-axis path does). Compares every real head against
one-process attention over the unpadded sequence.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

H, D, L = 56, 64, 1000


def attend(q, k, v, keep):  # (B, L, h, D), keys restricted to `keep`
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k[:, :, keep], v[:, :, keep])
    return out.transpose(1, 2)


def run(rank, world, ret):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29512")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from yunchang.comm.all_to_all import SeqAllToAll4D
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, L, H, D, generator=g, dtype=torch.float64) for _ in range(3))
    pad_l = -L % world
    padded = [torch.cat([t, torch.zeros(1, pad_l, H, D, dtype=t.dtype)], 1) for t in (q, k, v)]
    loc = [t.chunk(world, 1)[rank] for t in padded]               # rank's sequence shard
    pad_h = -H % world
    loc = [F.pad(t, (0, 0, 0, pad_h)) for t in loc]                # zero heads -> 57
    full = [SeqAllToAll4D.apply(None, t, 2, 1) for t in loc]       # whole sequence, 19 heads
    out = attend(*full, keep=torch.arange(L))                       # pad rows dropped as keys
    out = SeqAllToAll4D.apply(None, out, 1, 2)[:, :, :H]           # back to rows, drop dummy
    ref = attend(q, k, v, torch.arange(L)).chunk(world, 1) if pad_l == 0 else None
    ref_full = attend(q, k, v, torch.arange(L))
    ref_full = torch.cat([ref_full, torch.zeros(1, pad_l, H, D, dtype=ref_full.dtype)], 1)
    mine = ref_full.chunk(world, 1)[rank]
    real = slice(0, mine.shape[1] - (pad_l if rank == world - 1 else 0))
    err = (out[:, real] - mine[:, real]).abs().max().item()
    ret[rank] = (out.shape, err, bool(torch.isfinite(out).all()))
    dist.destroy_process_group()


if __name__ == "__main__":
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ret = mp.Manager().dict()
    mp.spawn(run, args=(world, ret), nprocs=world)
    for r in sorted(ret):
        print(f"world {world} rank {r}: out {tuple(ret[r][0])} max|err| {ret[r][1]:.3e} finite {ret[r][2]}")
