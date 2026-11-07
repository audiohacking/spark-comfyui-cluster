# Three Sparks

From two Sparks to three, 2026-11-06 to 2026-11-07: spark-a (driver +
worker), spark-b, spark-c.

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
- **2 Sparks, re-run of H3_00191 (seed 823985181201152):** latents
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
| A port 1 ↔ B port 0 | FABRIC_SUBNET/24, FABRIC_SUBNET/24 | A .2, B .1 |
| B port 1 ↔ C port 0 | FABRIC_SUBNET/24, FABRIC_SUBNET/24 | B .1, C .2 |
| C port 1 ↔ A port 0 | FABRIC_SUBNET/24, FABRIC_SUBNET/24 | C .2, A .1 |

Each Spark only knew its own two cables' subnets; the third cable's addresses
fell through to the default route on the management NIC (`enP7s7`, which
negotiated **1 GbE** on all three). `/etc/netplan/98-spark-fabric-routes.yaml`
on each Spark adds /32 routes to the four non-adjacent addresses via the
direct cable to their owner. In a 3-ring every pair is adjacent, so nothing is
forwarded. `cluster/rt.sh` checks all twelve addresses from a Spark.

Cluster addressing: each Spark is named by its port-1 address (A
`FABRIC_IP` = driver and NCCL master, B `FABRIC_IP`, C
`FABRIC_IP`), so `NCCL_SOCKET_IFNAME = enp1s0f1np1` finds the right
address everywhere. NCCL uses all four port halves (`NCCL_IB_HCA` lists
`rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`) with
`NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_IB_MERGE_NICS=0`. The image's NCCL
is 2.30.7.

VPN was not involved: its routing table (52) held only the VPN network /32s,
with `RouteAll` false and no exit node. Hostnames resolve to the LAN or
VPN network, so tests run by hostname never touch the fabric.

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
all-to-all), with 0 on the management NIC and on VPN.

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

**Aligned to A/B** (known good):
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
- **Hostname:** `spark-c` (was `spark-c`).

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
