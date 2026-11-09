# Three Sparks

From two Sparks to three, 2026-11-06 to 2026-11-08. The Sparks are called A
(driver and worker), B and C below.

## Ulysses at any degree (`container/monarch-ulysses-any-world.patch`)

dgx-monarch's Ulysses split needed the head count divisible by the number of
ranks, and `topology = auto` had only 2-rank rows (an odd world fell back to a
pure ring). The patch:

- **Head padding** (`adapters/base.py`, `pad_heads`): before the all-to-all,
  the head axis grows to a multiple of the degree with *copies of the leading
  real heads*, never zeros. Sage quantizes each head block by its own max, so
  an all-zero block risks 0/0. The copies are sliced off after the reverse
  all-to-all.
  - Exact for the real heads.
  - H3's 56 heads over 3 ranks become 57: about 2% more attention work.
- **Family refusals become no-ops:** `refuse_indivisible_heads` (flux3),
  `_assert_ulysses_divides_heads` (zimage) and `_assert_ring_only_topology`
  (omnigen2).
- **Auto topology:** world > 2, batch 1, and world > the model's parallel
  degree gives `ulysses = world`. World 2 is unchanged.

Tests (`docs/engineering/scripts/ulysses/`, CPU over gloo):
- **Exact:** head padding through yunchang's `SeqAllToAll4D` is exact (0 error
  in fp64), and through dgx-monarch's real USP wrapper at world 3 it stays
  within the same ≤8e-7 fp32 noise as world 2.
- **World-2 decisions unchanged** across the topology table.

On hardware:
- **2 Sparks, re-run of an earlier render with the same seed:** latents
  bit-identical to the day before. Main pass video `e8052715…` / audio
  `95b628af…`, face pass `eb088d72…` / `540b9ec0…`, at the same sampling
  times (302.6 s, 228.0 s).
- **3 Sparks:** `uly3`, all three ranks up, every proof render bit-identical
  across ranks. The identity gate passed in 319 s over 3 proof renders.

## The ring fabric

No switch: each Spark's port 0 is cabled to the next one's port 1. NVIDIA's
network setup (`/etc/netplan/99-nvidia-sync-cluster.yaml`, which disables older
fabric files as `*.sync-disabled-*`) gives every cable two /24s, one per
PCIe half of the port:

| Cable | Subnets | Addresses |
|---|---|---|
| A port 1 ↔ B port 0 | two /24s, one per PCIe half | one address per Spark per half |
| B port 1 ↔ C port 0 | two /24s | one address per Spark per half |
| C port 1 ↔ A port 0 | two /24s | one address per Spark per half |

Each Spark only knew its own two cables' subnets; the third cable's addresses
fell through to the default route on the management NIC (which
negotiated **1 GbE** on all three). `/etc/netplan/98-spark-fabric-routes.yaml`
on each Spark adds /32 routes to the four non-adjacent addresses via the
direct cable to their owner. In a 3-ring every pair is adjacent, so nothing is
forwarded. `cluster/rt.sh` checks all twelve addresses from a Spark.

Cluster addressing: each Spark is named by its port-1 address (A's is the
driver and NCCL master address), so `NCCL_SOCKET_IFNAME = enp1s0f1np1` finds the right
address everywhere. NCCL uses all four port halves (`NCCL_IB_HCA` lists
`rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`) with
`NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_IB_MERGE_NICS=0`. The image's NCCL
is 2.30.7.

A VPN overlay on the management side was not involved: its routes covered only
its own addresses. Hostnames resolve to the management network or the VPN, so
tests run by hostname never touch the fabric.

### Measured (perftest `ib_write_bw -R`, iperf2)

| Test | Result |
|---|---|
| RDMA, one port half | 92.4 Gb/s |
| RDMA, both halves of one cable together | 46.1 + 46.1 = 92 Gb/s (per cable) |
| RDMA, one half on each of two cables | 55.8 + 55.8 = 111 Gb/s |
| RDMA, all four halves (both neighbours) | 4 × 46.1 = **184.6 Gb/s** out of one Spark |
| TCP, 1 / 4 / 8 streams | 27.5 / 81.2 / 94.1 Gb/s |

The "26 Gb/s" a speed test showed was one TCP stream (one CPU core at MTU
1500), not the routing. During a 3-Spark H3 render, every Spark spread its
traffic evenly over all four halves (~2.6 Gb/s each, 10 s average, bursty
all-to-all), with 0 on the management NIC and the VPN.

## The host baseline

The third Spark came with a newer DGX OS (OTA 26.09.3 vs 26.09.2): kernel
7.0.0-1025, BIOS 5.36_0ACUM028, and `nvidia-spark-mlnx-firmware-manager`
6.0.1 enabled. That manager produced the "no cables" message during setup: it
refuses to flash without a cable.

**The crash.** During the second proof render of the first 3-Spark job,
at 14:49:00:
- an `mstflint` process oopsed in the `mstflint_access` module (NULL
  dereference in `pci_bus_read_config_dword`, from
  `_vendor_specific_sem`/`get_space_support_status`);
- then `kcompactd0` and `rcu_exp_par_gp` soft lockups froze the box. It still
  answered ping on the fabric, but nothing else.

dgx-monarch never calls `mstflint`; what launched it at that moment was not
identified. On the next boot the manager flashed the ConnectX-7 from
28.45.4028 to 28.50.1002.

**Aligned to A and B** (known good):
- **Kernel 7.0.0-1019-nvidia.** apt only offers its NVIDIA module as build
  `+2`, so A's exact `+1` build and the 1019 `hwe` meta packages were
  rebuilt from its installed files with `dpkg-repack`. They are kept in
  `~/kernel-1019-debs` on A and C. GRUB is pinned to the 1019 entry,
  and the meta packages are on hold.
- **ConnectX-7 28.45.4028:** the firmware manager 5.0.8's image, byte-identical
  to A's (sha256 `0a012e71…`, PSID NVD0000000087). It took two reboots:
  a pending image blocks a new burn until reset. Burns ran as detached
  `systemd-run` units, so an ssh drop could not interrupt them.
- **Firmware manager 5.0.8, disabled.**
- **Matching packages:** `nvidia-spark-initcall-bl` 1.0-1 (the boot options
  match again: `initcall_blacklist=tegra234_cbb_init`), `nvidia-modprobe` and
  `nvidia-settings` 615.71.09, `cuda-toolkit-config-common` 13.4.92, and
  `python3.12-dev`.
- **Tuning:** the GPU clock cap `-lgc 300,2300`, `comfyui-tune.service` (swap
  off, persistence mode), `vm.swappiness=10`.
- **Docker:** the classic image store
  (`{"features": {"containerd-snapshotter": false}}`). With the containerd
  store a pushed image gets another ID.
- **Hostname:** renamed to the cluster's naming scheme.

Pitfall: downgrading a library listed in the CDI spec
(`libnvidia-egl-wayland1`) left the worker container unable to start
(`failed to fulfil mount request … .so.1.1.23`). Running
`systemctl restart nvidia-cdi-refresh` regenerates the spec.

`cluster status` now compares every Spark's kernel, ConnectX firmware, Docker
image store, swap and firmware-manager state with the driver's.

## Performance (H3 ref2va, 5 s, 3 Sparks)

| Phase | Steady state (16:19 run, stock fp32 text encoder) |
|---|---|
| text encode, main pass | ~52 s |
| main pass sampling | 275.3 s |
| VAE decode (split over 3) | 14.9 s |
| text encode, face pass | 48 s |
| SAM3 face detection (124 frames) | ~10 s |
| face crop preparation | ~28 s |
| face pass sampling | 193.4 s |
| final decode and save | ~20 s |
| **total** | **646 s** |

Sampling is 72% of the job. The text encodes are now ~10–15 s each, and often
cached (see [text-encoders.md](text-encoders.md)).

Open question: the 2-Spark main pass took 302 s for 57 latent frames, and the
3-Spark one 276 s for 37. The 2-step proof renders took ~75 s on both.
Per-step time barely followed the shorter video or the extra Spark, so part of
each step's cost may not scale. Running the same job on 2 and on 3 Sparks
would settle it.

## Where a sampling step went, and the slow Spark (2026-11-07/08)

A one-step torch profile on every rank (patch `monarch-step-profile`, request
with `scripts/cluster/step_profile.sh`, read with `analyze_step_trace.py`) of a
face pass (39,621 rows, 13,207 per rank) showed every GPU 99.6-99.9% busy, so
nothing was launch-bound. Per DiT block (50 per step, ~385 ms):

| | ms per block |
|---|---|
| NCCL all-to-all (4 per block, Ulysses q/k/v/out) | 152-173 |
| Sage attention (int8 QK, fp8 PV) | 93-105 |
| int8 GEMMs (Triton, 4 per block) | 56-62 |
| copies, cat, transpose (Ulysses pack/unpack, 57-head padding) | 31-34 |
| quantize, norm, RoPE, elementwise | ~30 |

Compute (~10.7 s per step) matched a single-GB10 H3 profile scaled to our
per-rank shapes (a published single-GB10 H3 profile: attention ~164-180 TOPS on both). The
all-to-all moved 128 MB out of each rank in ~39 ms: 25 Gb/s, with no overlap.

`scripts/cluster/a2a_bench.py` (one process per Spark in the worker
containers, `a2a_bench.sh`) reproduced it outside a render: 41 ms at the
traced shape, linear in size (a bandwidth wall), and no NCCL setting moved it
(channels, QPs per connection, buffer and chunk sizes, protocol, GDR level).
Its p2p/bidir/ring modes then showed every transfer *from* Spark B at
~13-16 Gb/s and every other direction at 90-100 Gb/s; raw `ib_write_bw`,
`ib_read_bw` responses and TCP agreed. Cables, FEC, error and pause counters,
NIC firmware configuration (mstconfig), QoS, IOMMU mode, PCIe width/speed/MPS/
MRRS, memory bandwidth, clocks and temperatures were identical. The one
difference: Device Control 2 on B's NIC root ports read 0x0020 (LTR off)
against 0x0420 on the others. Secure Boot lockdown refuses `setpci`, so B
was rebooted: LTR on, 92.4 Gb/s RDMA and 91.5 Gb/s TCP out, all-to-all
41 -> 6.2 ms (166 Gb/s out per rank).

After that reboot:
- B's boot stalled on `plymouth-quit-wait`, so `comfyui-tune` (swap off)
  and `gb10-clock-cap` waited; `sudo plymouth quit` released them. Check
  `systemctl is-system-running` after rebooting a Spark.
- The next render failed at its first collective: `ibv_modify_qp failed with
  110 Connection timed out ... local GID <A's port-0 address>, remote GID
  <B's port-0 address>`. dgx-monarch's `generic-roce` profile pins
  `NCCL_IB_GID_INDEX=3`, but the RoCE v2 IPv4 GID sits at index 4 on some NICs
  (A's port-1 halves, C's port-0 halves), so NCCL paired NICs whose
  cables do not reach each other. `NCCL_IB_GID_INDEX = "-1"` in the fabric
  section lets NCCL choose per connection: 6.1 ms, every rank even.

## Fused small operations in the H3 block (`container/monarch-h3-block-fuse.patch`)

Stock `DiTBlock.forward` launches its RMSNorm, adaLN multiply and add, and the
gated residual as separate kernels, the modulation ones once per modulation
segment. The rank holding the text rows launches thousands of them per step.
Together they are 3.1-3.4% of a step's GPU time.

The patch runs the same arithmetic in three kernels per block with a per-row
modulation table, rounding to bf16 at the same points as stock and summing the
RMSNorm in PyTorch's order. On torch 2.14, 216 synthetic cases (widths 1024-5376,
1-600 rows, mixed integer and per-token segments, four input scales) matched
stock bit for bit, comparing the raw bf16 bits. A block's small operations took
0.22 ms against 0.57 ms at 600 rows.

Each Worker process repeats a smaller version of that comparison with its own
kernels on the first forward at a given width, and stays on stock if they differ
or do not compile. `DGXM_H3_BLOCK_FUSE=off` turns the path off.
