# Engineering notes

How the cluster's GB10-specific choices were measured and decided, with the
scripts that produced the numbers. The user-facing instructions are in the main
[README](../../README.md); this folder is the record behind them.

| Note | What it covers |
|---|---|
| [text-encoders.md](text-encoders.md) | bf16 tensor-core text encoders (mod `gb10-te-bf16`), the exact text-encoder cache, what was measured and ruled out |
| [cluster-3-spark.md](cluster-3-spark.md) | Scaling to three Sparks: Ulysses at any degree, the switchless ring fabric, the host baseline, and the third Spark's kernel crash |

## Scripts

`scripts/` holds the measurement and test scripts. They run inside the driver
container (ComfyUI at `/opt/ComfyUI`, the overlay at
`/opt/monarch-nodes/spark-continuity-monarch`), copied in with `docker cp`. The
GPU ones load the 32B H3 text encoder (~17 GB) and must run while no render is
queued.

| Script | What it does |
|---|---|
| `text-encoder/p0_inputs.py` | Shared input builder: an H3 ref2va presentation from `H3_PROMPT` (or `H3_PROMPT_FILE`), `H3_IMAGES` (`a.png\|b.png`) and `H3_VIDEO`, files in ComfyUI's `input/` |
| `text-encoder/p0_profile.py` | Encode time split into vision tower and LM layers, cold and warm, and run-to-run determinism |
| `text-encoder/p1_kernels.py` | torch.profiler table of the CUDA kernels one encode runs |
| `text-encoder/p2_invariance.py` | Whether a 3-way Ulysses split of an LM layer is bit-exact (`LAYERS=0,10,49`) |
| `text-encoder/p3_precision.py` | fp32 vs bf16 activations: warm time and drift |
| `text-encoder/p4_fp4probe.py` | Which kernels one nvfp4 linear runs, with and without full-precision matmul |
| `text-encoder/t_encode.py` | `save FILE` / `compare FILE`: one encode through ComfyUI's normal path, saved or compared bit for bit (`SPARK_TE_COMPUTE` decides the precision) |
| `text-encoder/p8_cachekey.py` | Whether the real H3 encoder gets a text-encoder cache key, and if not why |
| `text-encoder/test_te_cache.py` | CPU unit tests of `te_cache.py` (run with `CUDA_VISIBLE_DEVICES=` and the module path as argument) |
| `ulysses/t_headpad.py`, `t_wrapper.py`, `t_topo.py` | CPU tests of head padding through yunchang's all-to-all and dgx-monarch's real USP wrapper, and the topology decision per world size |
| `cluster/rt.sh` | On one Spark: every fabric address answers, through a fabric port |
| `cluster/ctr.sh` | Per-port RDMA and management traffic over 10 s (run on each Spark during a render) |
| `cluster/eq.sh` | One host's kernel, driver, Docker, swap, sysctl, memlock, repo, config and image, for side-by-side comparison |
| `cluster/therm.sh` | One host's boot options, clocks, modprobe options, NVIDIA services, BIOS and NVIDIA packages |
| `cluster/sync03.sh` | The fabric copy that seeded the third Spark (repos, data, models; parallel rsync) |
