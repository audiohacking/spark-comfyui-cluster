"""dgx-monarch's real USP attention wrapper at Ulysses degrees that do not divide
the head count, on CPU ranks over gloo with xfuser's own process groups.

Per rank: its sequence shard of a padded stream (H3-like: 56 heads, packed
length not a multiple of the degree, pad rows passed as drop_rows the way the
adapter passes them). Compared with one-process attention over the unpadded
stream. TORCH_MATH so CPU can run it; the wrapper logic is kernel independent.
"""
import os
import sys

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

H, D, L = int(os.environ.get("H", 56)), 64, 1000


def run(rank, world, ret):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29600 + world),
                      RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0")
    torch.cuda.set_device = lambda *a, **k: None  # CPU ranks: xfuser only pins a device
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(world_size=world, rank=rank, backend="gloo")
    initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world,
                              ring_degree=1, backend="gloo")
    from dgx_monarch.adapters.base import make_usp_attention, ulysses_world
    attn = make_usp_attention("TORCH_MATH")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, L, H, D, generator=g) for _ in range(3))
    pad = -L % world
    drop = list(range(L, L + pad))
    full = [torch.cat([t, torch.zeros(1, pad, H, D)], 1) for t in (q, k, v)]
    loc = [t.chunk(world, 1)[rank].contiguous() for t in full]
    ref = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
    ref = torch.cat([ref, torch.zeros(1, pad, H, D)], 1).chunk(world, 1)[rank]
    n = ref.shape[1] - (pad if rank == world - 1 else 0)
    res = {}
    # (B, L, H*D) form and (B, H, L, D) skip_reshape form, both as comfy calls them.
    o1 = attn(*(t.reshape(1, -1, H * D) for t in loc), H, drop_rows=drop or None)
    res["flat"] = (o1.reshape(1, -1, H, D)[:, :n] - ref[:, :n]).abs().max().item()
    o2 = attn(*(t.transpose(1, 2) for t in loc), H, skip_reshape=True, drop_rows=drop or None)
    res["bhld"] = (o2.reshape(1, -1, H, D)[:, :n] - ref[:, :n]).abs().max().item()
    ret[rank] = (ulysses_world(), tuple(o1.shape), res)


if __name__ == "__main__":
    world = int(sys.argv[1])
    ret = mp.Manager().dict()
    mp.spawn(run, args=(world, ret), nprocs=world)
    for r in sorted(ret):
        u, shape, res = ret[r]
        print(f"H={H} world {world} rank {r}: ulysses {u} out {shape} "
              + " ".join(f"{k} max|err| {v:.2e}" for k, v in res.items()))
