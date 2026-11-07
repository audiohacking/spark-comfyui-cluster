"""All-to-all bandwidth at the Ulysses shape H3 sends, over the fabric NCCL uses.

Run one process per Spark inside its worker container (a2a_bench.sh does that),
with RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT and the NCCL env set. It times
torch.distributed.all_to_all_single on a [world, rows, 1, heads, 128] bf16
buffer, the exact call yunchang's SeqAllToAll4D makes (c10d::alltoall_base_).
Env: A2A_MODE (a2a | p2p: A2A_SRC -> A2A_DST (0 -> 1) | bidir: SRC <-> DST | ring: r -> r+1), A2A_ROWS (13207), A2A_HEADS (19), A2A_ITERS (20), A2A_WARMUP (5).
"""
import os
import time

import torch
import torch.distributed as dist

rows = int(os.environ.get("A2A_ROWS", "13207"))
heads = int(os.environ.get("A2A_HEADS", "19"))
iters = int(os.environ.get("A2A_ITERS", "20"))
warmup = int(os.environ.get("A2A_WARMUP", "5"))
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)
x = torch.randn(world, rows, 1, heads, 128, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)
mode = os.environ.get("A2A_MODE", "a2a")   # a2a | p2p (rank 0 -> 1) | bidir (0 <-> 1) | ring (r -> r+1)


def op():
    if mode == "a2a":
        dist.all_to_all_single(y, x)
        return
    reqs = []
    if mode == "ring":                     # every rank sends right and receives from the left
        right, left = (rank + 1) % world, (rank - 1) % world
        # batched, so it runs on this 3-rank communicator (unbatched pairs deadlock)
        ops = [dist.P2POp(dist.isend, x[right], right), dist.P2POp(dist.irecv, y[left], left)]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        return
    src, dst = int(os.environ.get("A2A_SRC", "0")), int(os.environ.get("A2A_DST", "1"))
    ops = []
    if rank == src:
        ops.append(dist.P2POp(dist.isend, x[dst], dst))
        if mode == "bidir":
            ops.append(dist.P2POp(dist.irecv, y[dst], dst))
    elif rank == dst:
        ops.append(dist.P2POp(dist.irecv, y[src], src))
        if mode == "bidir":
            ops.append(dist.P2POp(dist.isend, x[src], src))
    for r in (dist.batch_isend_irecv(ops) if ops else []):
        r.wait()


for _ in range(warmup):
    op()
torch.cuda.synchronize()
dist.barrier()
times = []
for _ in range(iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dist.barrier()
    start.record()
    op()
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
# correctness: chunk j of what rank r receives is what rank j sent to r
ref = torch.empty_like(x)
dist.all_to_all_single(ref, x)
torch.cuda.synchronize()
sent = x[0].numel() * x.element_size() * ((world - 1) if mode == "a2a" else 1)   # bytes leaving a sending rank
times.sort()
med = times[len(times) // 2]
every = [torch.zeros(1, device="cuda") for _ in range(world)]
dist.all_gather(every, torch.tensor([med], device="cuda"))
t = max(every, key=lambda v: v.item())
if rank == 0:
    print(f"{mode} {tuple(x.shape)} {x.numel() * 2 / 1e6:.1f} MB buffer, {sent / 1e6:.1f} MB out per rank: "
          f"median {med:.2f} ms (slowest rank {t.item():.2f} ms) min {times[0]:.2f} -> "
          f"{sent * 8 / (t.item() / 1000) / 1e9:.1f} Gb/s out per rank; per rank "
          f"{[round(v.item(), 2) for v in every]} ms", flush=True)
dist.destroy_process_group()
