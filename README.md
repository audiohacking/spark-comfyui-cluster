<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.svg">
  <img alt="spark-comfyui — ComfyUI on the NVIDIA DGX Spark (GB10 Grace Blackwell)" src="docs/banner-light.svg" width="760">
</picture>

**One script to install, run, update, and maintain [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on the NVIDIA DGX Spark (GB10 Grace Blackwell), fully containerized.**

[![CI](https://github.com/audiohacking/spark-comfyui-cluster/actions/workflows/ci.yml/badge.svg)](https://github.com/audiohacking/spark-comfyui-cluster/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/audiohacking/spark-comfyui-cluster?color=76B900)](LICENSE)
![Platform](https://img.shields.io/badge/platform-DGX_Spark_%C2%B7_GB10_%C2%B7_sm__121-76B900)
![Cluster](https://img.shields.io/badge/cluster-dgx--monarch-76B900)

A hard fork of [bjarkebolding/spark-comfyui](https://github.com/bjarkebolding/spark-comfyui) that adds multi-Spark clustering. See [About this fork](#about-this-fork).

</div>

- **One image, whole stack.** cu130 PyTorch, native sm_121 SageAttention, GPU onnxruntime and the GB10 mods, baked in and reproducible from this repo.
- **Your content stays plain.** Models, workflows, custom nodes and outputs live in a `data/` directory next to the script, bind-mounted in. No venv, no system Python changes.
- **Third-party code runs confined.** Custom nodes are arbitrary Python. They run non-root, with all capabilities dropped, seeing nothing but your content and the GPU.
- **Several Sparks, one render** *(this fork)*. [dgx-monarch](https://github.com/Deen-Media/dgx-monarch) is baked into an optional image stage, so one ComfyUI window can split a render (MiniMax H3, Continuity) across every Spark on the fabric. Off until you create one config file.

## About this fork

This fork tracks upstream [bjarkebolding/spark-comfyui](https://github.com/bjarkebolding/spark-comfyui) and adds clustering on top of it. Without a cluster config it behaves exactly like upstream: the same image, the same container flags, the same commands. The stand-alone `runtime` stage of the Dockerfile is unchanged, so a stand-alone build gives the image upstream would.

What the fork adds:

| Area | Upstream | This fork |
|---|---|---|
| Image | One `runtime` image | Same, plus an optional `cluster` stage with dgx-monarch (newest release tag) and Continuity's cluster support. Built only when `spark-cluster.toml` exists; bumping dgx-monarch rebuilds one thin layer. |
| Container roles | ComfyUI only | `SPARK_ROLE=driver` (ComfyUI, the default) or `worker` (a dgx-monarch worker loop). `SPARK_MONARCH=1` loads the dgx-monarch node pack from the image, not from your `custom_nodes`. |
| Networking | Bridge network, `-p PORT:8188` | Unchanged stand-alone. Cluster mode uses `--network host --pid host`, the RDMA devices and an unlimited memlock limit, because Monarch and NCCL need the fabric and dgx-monarch's probes read the host's `/proc`. |
| Default nodes | — | [Continuity](https://github.com/roadmaus/ComfyUI-Continuity) is installed on every start when missing, and can be updated from ComfyUI Manager. See [Bundled by default: Continuity](#bundled-by-default-continuity). In cluster mode its H3 renders sample across the Sparks ([Continuity on the cluster](#continuity-on-the-cluster)). |
| Commands | — | `cluster init \| worker \| push \| status \| dgxm`, see [Clustering](#clustering-dgx-monarch). |
| Upgrades | `update` / `--rollback` / `--keep` for ComfyUI | The same commands also move and snapshot dgx-monarch (`org.spark-comfyui.monarch-sha` label). `cluster push` syncs the result to every Spark. |
| Several installs per box | One per Spark (fixed names, port 8188) | UI on port **8182** by default, so it never lands on a stock install's 8188. A per-folder `spark-comfyui.env` sets the container, image and cache volume names (and the port), and `run`/`service` refuse a port someone else holds. See [Running two installs on one Spark](#running-two-installs-on-one-spark). |

### Getting it

```bash
git clone https://github.com/audiohacking/spark-comfyui-cluster.git
cd spark-comfyui-cluster
```

Clustering is on `main`. The default clone folder, `spark-comfyui-cluster`, already differs from a stock checkout's `spark-comfyui`; give it its own names and port as below. Then follow the [Quick start](#quick-start), or [Quick test: one Spark, then two](#quick-test-one-spark-then-two). `update` self-updates from this fork's `main`, not from upstream.

### Next to an existing upstream install

Point a second checkout at its own names before the first `install`. The stock install keeps its container, image, cache volume and port 8188. This fork's UI is already on 8182 by default:

```bash
cat > spark-comfyui.env <<'EOF'
CONTAINER_NAME=spark-comfyui-cluster
CONTAINER_IMAGE=spark-comfyui-cluster
PORT=8182
EOF
./spark-comfyui.sh status    # "instance:" shows the names in effect
```

Two things stay shared on purpose:
- **The BuildKit cache.** Both installs reuse the torch and SageAttention layers. While the other install's images exist, this fork leaves trimming the cache to that install.
- **`tune`.** It's host-wide, and `tune --revert` refuses while another install exists.

Only one dgx-monarch worker can run per Spark.

**Don't download your models again.** Point the new install at the existing `data/` before its first `run`. See [Sharing an existing data folder](#sharing-an-existing-data-folder).

### Following upstream

This is a hard fork, and nothing goes back upstream. Upstream fixes still come in when you choose to take them:

```bash
git remote add upstream https://github.com/bjarkebolding/spark-comfyui.git
git fetch upstream && git merge upstream/main
```

Fork-only code is kept in separate files and functions, so merges rarely conflict.

## Contents

[About this fork](#about-this-fork) · [Quick start](#quick-start) · [Commands](#commands) · [Configuration](#configuration) · [What it looks like](#what-it-looks-like) · [Mounts](#mounts) · [Patch list](#patch-list-optional) · [Node list](#node-list) · [Recipes](#recipes) · [Clustering](#clustering-dgx-monarch) · [Troubleshooting](#troubleshooting) · [Security notes](#security-notes)

## Quick start

Needs a DGX Spark (GB10) on DGX OS (docker, the NVIDIA Container Toolkit and the r580 driver ship with it), plus ~25 GB of disk before models.

```bash
git clone https://github.com/audiohacking/spark-comfyui-cluster.git
cd spark-comfyui-cluster
./spark-comfyui.sh install          # one-time image build, about 5 min
./spark-comfyui.sh tune --persist   # recommended: swap off, persistence mode
./spark-comfyui.sh run              # UI at http://<spark-ip>:8182
```

Models go in `data/models/checkpoints` (etc.). No venv, no system Python changes; `sudo` is only used by `tune`.

## Commands

| Command | What it does |
|---|---|
| `install` | Builds the image: ComfyUI at a pinned commit, cu130 PyTorch, SageAttention compiled for sm_121, GPU onnxruntime (sha256-pinned), GB10 mods. Idempotent. |
| `run [args...]` | Starts ComfyUI in the container, foreground. Every start installs custom-node requirements and live-verifies the GPU stack before serving. Extra args pass to `main.py`. |
| `service [args...] \| --disable` | Same, detached with a docker restart policy: survives crashes and reboots. Extra args pass to `main.py` as with `run`, and stay in effect across restarts until you `stop` and start the service again; `status` shows them. `docker ps` reports `healthy` once the server actually answers, not merely once the process exists. |
| `stop` | Stops ComfyUI. |
| `update [--torch] [--rollback] [--keep[=NAME]]` | Self-updates the tool, then rebuilds the image on current ComfyUI master (cached layers: minutes). The old image stays as `:previous`; `--rollback` swaps back instantly. `--torch` forces fresh PyTorch wheels. `--keep` also pins the image you are leaving as `:keep-NAME` (default: today's date), which survives every later update and `prune`. |
| `doctor [--cold]` | Health check: tool and host (driver, docker, image, swap, backups), then the live GPU gates (torch CUDA, sm_121 SageAttention kernel, onnxruntime provider, NVFP4) inside a throwaway container. Every failure names its fix. `--cold` additionally rebuilds the ComfyUI stage against upstream master as it stands right now, which is what a fresh install would get and what your cached layers otherwise hide. Adds about 3 minutes. |
| `status [--watch [SEC]]` | One-page glance, or a live sparkline dashboard with generation telemetry and a `session:` A/B summary. Every sample lands in `thermal_monitor.log`, the post-mortem trail for silent hard-reboots. |
| `tune [--clock-cap MHZ] [--persist]` <br> `tune --revert` | Host stability: swap off, persistence mode, optional clock cap (~2100 fixes overcurrent hard-reboots). `--revert` puts all of it back: swap on, clocks unlocked, persistence off, the boot unit and the sysctl line removed. |
| `backup [--with-output] [FILE]` | Small tgz of workflows, settings, inputs, configs and the custom-node set. Models are manifested, never archived. Safe while running. |
| `restore FILE` | Rebuilds from a backup: image if missing, content into `data/`, custom nodes re-cloned at pinned commits, missing models listed with sizes. |
| `prune [--yes]` | Reclaims disk: drops leftover image tags and trims the BuildKit cache to its age and size limits. Keeps `:latest`, `:previous` and every `keep-*` pin, and never rebuilds. Shows what goes before it goes. |
| `reset [--yes]` | Removes the container, all image tags and the cache volume; rebuilds from scratch. `data/` is never touched. Unlike `prune`, this does drop `keep-*` pins. |
| `recipe list\|show\|check\|install\|capture` | Everything a workflow needs, as data: the workflow, its custom nodes, and every model with a destination, size and sha256. `install` downloads what is missing (resumable, hash-verified) and wires up the rest. `capture` authors one from a workflow that already runs here. |
| `cluster init\|worker\|push\|status\|dgxm` | Splits renders across several Sparks with [dgx-monarch](https://github.com/Deen-Media/dgx-monarch). Off until `spark-cluster.toml` exists. See [Clustering](#clustering-dgx-monarch). |
| `shell` | Opens a bash shell inside the running container. For inspecting the venv or a custom node. The image is immutable, so nothing you change there survives a restart. |

## Configuration

Everything is an environment variable, so nothing needs editing to try something.

### Running two installs on one Spark

The default names are upstream's: container and image `spark-comfyui`, cache volume `spark-comfyui-cache`. Only the port differs, since this fork's default is `8182` and upstream's is `8188`. A second checkout next to a stock install needs its own names. Put them in `spark-comfyui.env` next to the script, which is gitignored and read on every command:

```bash
CONTAINER_NAME=spark-comfyui-cluster
CONTAINER_IMAGE=spark-comfyui-cluster   # also names the cache volume: spark-comfyui-cluster-cache
PORT=8182
```

The file holds plain `KEY=VALUE` lines. It is parsed, not sourced, and a variable already set in the environment wins. The worker container follows `CONTAINER_NAME` (`spark-comfyui-cluster-worker`). `status` prints the names in effect, and `run`/`service` refuse a port another container or process already holds, naming who has it. `SPARK_ENV_FILE=PATH` points at a different file.

Everything this checkout creates or acts on follows those names, so the two installs never touch each other:

| Thing | This fork (with the file above) | Stock install |
|---|---|---|
| Containers | `spark-comfyui-cluster`, `spark-comfyui-cluster-worker` | `spark-comfyui` |
| Images and tags | `spark-comfyui-cluster:latest`, `:previous`, `:keep-*` | `spark-comfyui:*` |
| Cache volume | `spark-comfyui-cluster-cache` | `spark-comfyui-cache` |
| UI port | 8182 (cluster: also 26600 worker, 29777 NCCL) | 8188 |
| `status`, `stop`, monitor | this instance's container (or host process) only | its own |
| `prune` of untagged images | images labelled `org.spark-comfyui.instance=spark-comfyui-cluster` | its own |
| Boot tuning unit | `spark-comfyui-cluster-tune.service` | `comfyui-tune.service` |
| Cluster ssh key | `~/.ssh/spark-comfyui-cluster/` | none |
| dgx-monarch host state | `~/.local/state/spark-comfyui-cluster/` | none |
| Host python shim | `~/.local/share/spark-comfyui-cluster/python` | none |
| Content | `data/` in this checkout, or what `spark-mounts.conf` names | its own `data/` |

The pieces of the host itself are shared because there is only one of each:
- **The BuildKit cache.** It is left to the other install to trim, as noted under the quick test.
- **The GPU and system tuning** (swap, clocks). `tune --revert` refuses while another install exists, because reverting would undo that install's tuning too.
- **The models, if you choose to share them** ([Sharing an existing data folder](#sharing-an-existing-data-folder)).

### Runtime

Set at `run` or `service` time.

| Variable | Default | What it does |
|---|---|---|
| `SPARK_ATTENTION` | `sage` | Attention backend: `sage`, `sdpa`, or `ck` for upstream's Comfy Kitchen INT8. Whichever you pick is verified with a real kernel run before the server starts, and a failure refuses to launch rather than quietly serving on another backend. |
| `SPARK_RESERVE_VRAM` | unset | Keeps that many GB of the unified pool free. Hardens against the overcommit freeze when pushing large models, and leaves room for a co-resident process. |
| `SPARK_BF16` | `1` | The bf16 fast path. `0` disables it. |
| `SPARK_BF16_VAE` | `1` | Keeps the fast path but takes the VAE off bf16 when set to `0`. |
| `SPARK_STATIC_VRAM` | `0` | `1` disables DynamicVRAM. |
| `SHM_SIZE` | `16g` | The container's `/dev/shm` ceiling. The Spark figure for large tensor transfers. |
| `PORT` | `8182` | Host port. Upstream uses 8188; this fork moves off it so both can run on one Spark. Inside the container ComfyUI still listens on 8188. |
| `BIND_ADDR` | unset | Host interface the port is published on. Empty is docker's own behaviour and means every interface, because a Spark is a headless box you reach from a laptop. |

```bash
BIND_ADDR=127.0.0.1 ./spark-comfyui.sh service   # this box only
SPARK_ATTENTION=ck ./spark-comfyui.sh run        # try Comfy Kitchen INT8
```

`status` reads the binding back from the running container rather than from your environment, so it always prints a URL that actually works. Reaching a loopback-bound UI from elsewhere is then an SSH tunnel: `ssh -N -L 8182:127.0.0.1:8182 you@spark`.

> [!NOTE]
> On this hardware the attention backend only matters once the workload is compute-bound. At 1024x1024 all three land inside the noise band; at 1536x1536 SDPA costs about 4 percent while `sage` and `ck` tie.

The VAE one has a flag form, `--no-bf16-vae`, which works with any command:

```bash
./spark-comfyui.sh service --no-bf16-vae          # LTX-2.3 audio workflows
./spark-comfyui.sh run --no-bf16-vae --fp32-vae   # or pick your own precision
```

### Build pins

| Variable | Default | What it does |
|---|---|---|
| `TORCH_VERSION` | `2.14.0` | Pinned so a PyTorch release can never change your stack without you asking. |
| `TORCHVISION_VERSION` | `0.29.0` | Pinned alongside torch. |
| `TORCHAUDIO_VERSION` | `2.11.0` | Pinned alongside torch. |

`doctor` tells you when a newer torch is on the index, and `update --torch` builds against it, ignoring the pin. If that passes `doctor` and a generation, move the pin.

### Disk

Applied by `update` and `prune`. Either at `0` disables that pass.

| Variable | Default | What it does |
|---|---|---|
| `CACHE_KEEP_DAYS` | `7` | Drops build cache untouched for that long. |
| `CACHE_MAX_GB` | `40` | Caps total build cache size. |

The cap is the one that matters if you rebuild often, because the age filter measures last use and a frequent rebuild keeps every layer fresh.

## What it looks like

`status --watch` during a run. Heat-colored timeseries; rows appear only when they carry information:

```console
$ ./spark-comfyui.sh status --watch 3

  ─ GPU ──────────────────────────────────────────────────────────────────────
  temp         57°C ↗                   ▁▁▁▃▅▅▅▁▁▁▅▆▆▆▁▁▁▁▅▅▆█▃▁▁  46–61 ~53
  power      77.40W ↗                   ▁▁▁▇███▁▁▁████▁▁▁▁███▅▁▁▁  10.4–77.4 ~32.5
  sm clk    2411MHz                     ▂▂▅█▆▆▆▂▂▂▆▅▅▅▂▂▂▂▅▅▅▅▂▂▂  2398–2463 ~2421 P0
  gpu           96%                     ▁▁▄▇███▁▁▁████▁▁▁▁████▁▁▁  0–96 ~33.6
  ─ SYSTEM ───────────────────────────────────────────────────────────────────
  unified     22.7G                     ▁▁▃██████████████████████  4.1–23.6 ~20.9 of 122G
  cpu            5%                      ▁▃█▅▆▅▃▁▁▄▅▅▅▂▁▁▁▅▅▅▆▁▁▁  1–10 ~3.1
  disk io   0.0MB/s                      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁  0–0.4 ~0.04
  ─ GENERATION · SageAttention ───────────────────────────────────────────────
  gen         13.1s                             ███████▁▁▁▁▁▁▁▁▁▁  13.1–29.8 ~17.3
  it/s         0.63                         ▁▃▃▄   █▃▃▄    █▃▃     0.4–1.36 ~0.72
  latency       13s                             ███████▅▅▅▅▅▅▅▁▁▁  13–30 ~17.4
  queue           0                     ▁▁▁█████▁▁▁████▁▁▁▁███▁▁▁  0–1 ~0.34
  hit rate      77%                             ▁▁▁▁▁▁▁██████████  0–77 ~57.8
  session: 4 gens · first 29.8s · steady ~13.6s (13.6–13.6) · ~0.63 it/s
```

> [!TIP]
> The `session:` line is made for A/B tests: run one watch per condition and compare. That is how the container itself was validated: native steady 13.59s vs container 13.61s on the same workflow and seeds, every seed-matched output pair bit-identical.

A healthy `doctor` looks like this. The gates run inside a throwaway container from the exact image `run` uses:

<div align="center">
  <img alt="spark-comfyui.sh doctor: all GPU gates pass" src="docs/doctor.svg" width="720">
</div>

## Mounts

Everything lives under `data/` by default. To relocate entries or add extra mounts, edit `spark-mounts.conf` next to the script (a commented template is seeded on install):

```
models = /mnt/fast-ssd/models
output = /mnt/nas/comfyui-output
mount = /mnt/nas/sdxl-models:/opt/ComfyUI/models/nas:ro
```

`status` always prints the resolved table. A typo in a `mount =` line dies loudly instead of silently shadowing your data with an empty directory. Point `extra_model_paths.yaml` entries at the container side of `mount =` lines.

Absolute paths are used as given and `~` is expanded. A configured directory does not have to exist yet, but its parent must, so a typo or an unmounted share fails loudly instead of quietly mounting an empty directory. Relative paths resolve against the config file's own directory.

For several setups, keep one file each and pick one per invocation with `--mounts`:

```bash
./spark-comfyui.sh --mounts ./nas-profile.conf run
./spark-comfyui.sh --mounts ./nas-profile.conf status   # check what it resolves to
```

The flag works with every command and must name a file that exists, so a typo fails loudly instead of quietly mounting the defaults. `MOUNTS_CONF=PATH` does the same thing as an environment variable. Use it on `backup` and `restore` too so they act on that setup. To run two setups at once, give each its own `CONTAINER_NAME`, `CONTAINER_IMAGE` and `PORT` (see [Running two installs on one Spark](#running-two-installs-on-one-spark)).

### Sharing an existing data folder

An existing install's content lives in its own `data/`: `models`, `custom_nodes`, `user` (settings and saved workflows), `input`, `output`, and optionally `extra_model_paths.yaml`. A second install on the same Spark doesn't need copies of any of it. Map what you want to share in the new checkout's `spark-mounts.conf`, **before its first `run`**. Otherwise the new install starts with an empty `data/`, and anything you queue downloads models into it again.

Models are the part that matters: hundreds of GB that you never want twice. Below, `OLD` is the existing install's data folder, for example `~/Documents/ComfyUI/spark-comfyui/data`.

**Recommended: share models, give each install its own nodes and settings.**

```ini
# spark-mounts.conf in the NEW checkout
models = ~/Documents/ComfyUI/spark-comfyui/data/models
# also share the yaml if the old install has one (it points at extra model locations)
# extra_model_paths.yaml = ~/Documents/ComfyUI/spark-comfyui/data/extra_model_paths.yaml
# shared inputs/outputs are optional and harmless:
# input  = ~/Documents/ComfyUI/spark-comfyui/data/input
# output = ~/Documents/ComfyUI/spark-comfyui/data/output
```

Then seed the new install's nodes and workflows once from the old one. It's a copy, so updating a node in one install never changes the other:

```bash
OLD=~/Documents/ComfyUI/spark-comfyui/data
mkdir -p data
cp -a "$OLD/custom_nodes" "$OLD/user" data/
cp "$OLD/../comfyui-nodes.list" . 2>/dev/null   # the same registry node list, if you use one
./spark-comfyui.sh status                         # the mounts table must show models -> $OLD/models
```

`status` prints the resolved table on every run. Check that `models` points at the old folder before you start. Model files arrive through the mount, so nothing is copied and both installs see a model downloaded by either one.

**Alternative: share everything.** Set `DATA_DIR` in `spark-comfyui.env` and the new install uses the old `data/` as its own:

```bash
echo "DATA_DIR=$HOME/Documents/ComfyUI/spark-comfyui/data" >> spark-comfyui.env
```

A `DATA_DIR` you set is treated as existing content that this checkout doesn't own:
- **It must already exist.** Every command checks first and stops if it's missing. It is never created: a typo or an unmounted disk can't quietly turn into an empty data folder.
- **Nothing is created inside it.** If an entry such as `user` or `custom_nodes` is missing, the command stops and names it instead of making an empty one. `extra_model_paths.yaml` is mounted only if the file exists.
- **The default `data/`** (no `DATA_DIR` set) still creates its skeleton on first use, as before.

That works, but both installs then run the same custom node code and write the same settings. Every start reconciles `comfyui-nodes.list` into that shared `custom_nodes`, and a node updated through Manager in one install changes under the other while it runs. Use it when the two don't run at the same time, or start them one at a time. A node's Python requirements are not shared: each container installs them into its own image environment on start.

**Things to know**

- **Ownership:** both installs run as whoever launches them, so use the same Linux user for both. `doctor` names any directory owned by another uid.
- **Sharing `user/`** (with `DATA_DIR`): startup prints `Another ComfyUI is already using this install's asset database` while the other install runs. It's harmless.
- **Configured paths must exist:** a configured path's parent must exist, so a typo or an unmounted share fails loudly instead of mounting an empty folder over your models.
- **Data commands:**
  - `reset` and `prune` never touch content, shared or not.
  - `restore` writes into whatever is mapped, including a shared folder.
  - `backup` manifests models and never archives them, so a shared `models` doesn't make backups bigger.
- **Clustering:** every Spark needs the same models at the same relative paths. Each Spark has its own disk, so copy just the files a workflow needs once over the fabric. Don't download them again:

  ```bash
  # from A: send A's diffusion models to B over the fabric, resumable
  rsync -a --partial --info=progress2 ~/Documents/ComfyUI/spark-comfyui/data/models/diffusion_models/ \
        192.168.100.12:~/Documents/ComfyUI/spark-comfyui/data/models/diffusion_models/
  ```

  A network share mapped with `models = /mnt/share/models` on every Spark works too, but each rank then loads the weights over the network.

## Patch list (optional)

`comfyui-patches.list` next to the script merges upstream PRs or fork branches (`pr:12345`, `branch:name`, `remote:<url> <branch>`) on top of ComfyUI inside the image build. A conflict fails the build loudly. Empty list means plain master tracking.

## Node list

`comfyui-nodes.list` next to the script declares which [Comfy Registry](https://registry.comfy.org) nodes you want. The container installs them on every start, one per line, in order:

```
comfyui-workflow-models-downloader     # latest release
some-node-id@1.2.3                     # pinned version
some-node-id@nightly                   # git HEAD
https://github.com/user/repo.git       # direct clone
```

It is the run-time counterpart to the patch list. Patches are a build input baked into the image; custom nodes live in the bind-mounted `custom_nodes` directory, so they can only be reconciled at launch. Manager's own `cm-cli` does the install, which is why every format it understands works here.

Nodes already on disk are skipped without a network call, so a normal start costs nothing. A node that fails to install warns and the server still starts, and the next start retries it. Removing a line stops future installs; it does not uninstall a node you already have.

`install` and `update` both seed the file with one active entry, `comfyui-workflow-models-downloader`. Comment it out if you would rather start with nothing.

### Bundled by default: Continuity

Every install of this fork gets [Continuity](https://github.com/roadmaus/ComfyUI-Continuity), the one-node MiniMax H3 / LTX / Krea front end, without editing anything. It comes from `container/default-nodes.list`, which is baked into the image and reconciled on every start right after your own list, with the same installer and the same rules:

- **Missing:** installed at its latest release on the next start.
- **Already there:** left alone, and found by folder name, pyproject name, git remote or declared repository. That covers a git clone from before, or one in a shared `custom_nodes`.
- **Updating:** use **Manager → Custom Nodes Manager → Continuity → Update** in the ComfyUI UI. The node lives in your `custom_nodes`, so the update persists across restarts and image updates, and the default list never downgrades it.
- **Pinning a version:** put a pinned entry for it in your own `comfyui-nodes.list`.
- **Opting out:** `SPARK_DEFAULT_NODES=0` skips the list. Uninstalling it through Manager only lasts until the next start, because the list then installs it again.
- **Changing the defaults:** edit `container/default-nodes.list` in the repo and run `update`. The change reaches every Spark with the image.

## Recipes

A workflow file does not say what it needs. It names models by bare filename, with no path, no URL and no hash, and the destination folder is only implied by the loader node, which breaks entirely for subgraphs. A recipe says all of it.

```bash
./spark-comfyui.sh recipe list                            # what is available
./spark-comfyui.sh recipe check image_krea2_turbo_t2i     # what is missing, downloads nothing
./spark-comfyui.sh recipe install image_krea2_turbo_t2i
```

```console
$ ./spark-comfyui.sh recipe check image_krea2_turbo_t2i

== image_krea2_turbo_t2i ==
  [want] loras/krea2_darkbrush.safetensors (448M)
  [want] diffusion_models/krea2_turbo_fp8_scaled.safetensors (13G)
  [want] text_encoders/qwen3vl_4b_fp8_scaled.safetensors (4.9G)
  [want] vae/qwen_image_vae.safetensors (243M)

  Krea-2 Turbo text-to-image
  models: 0 of 4 present, 4 to download (18G)
  [info] sizes checked, not hashes — full verification: ./spark-comfyui.sh recipe check image_krea2_turbo_t2i --verify

  [info] checking 4 source(s)

  Install it: ./spark-comfyui.sh recipe install image_krea2_turbo_t2i
```

Models already on disk show as `[have]`. The `checking N source(s)` step is a preflight: one HEAD request per missing model, which catches a gated file, a rotted URL or a recipe whose sizes have drifted before a single byte is downloaded. `check` reports them; `install` refuses to start.

`install` downloads what is missing, merges the recipe's custom nodes into `comfyui-nodes.list`, and drops the workflow into your workflows directory. Downloads resume, every model is checked against its sha256, and a file only moves into `models/` after it verifies, so a partial download can never look installed. A model already on disk that disagrees with the recipe is reported and **left alone**; `--force` re-downloads it.

Each recipe is a single self-contained `recipes/<name>.json` with the workflow embedded, so you can paste one into a gist or an issue and someone else can install from it. They are tracked in this repo, which makes a recipe the executable form of a forum post. Gated downloads read `HF_TOKEN` and `CIVITAI_TOKEN` from the environment; a token never belongs in a recipe.

To author one from a workflow that already runs on your box, name the workflow as you saved it:

```bash
./spark-comfyui.sh recipe capture my_workflow
```

A path to a `.json` file works too, and `--name` overrides the recipe name, which otherwise matches the workflow.

It finds each model under `models/`, records the destination, size and sha256, and fills in the URLs. Those come from recipes you already have, matched on sha256 so a URL is only reused for provably identical bytes, and failing that from ComfyUI-Manager's model list. Anything still blank you fill in yourself; a recipe with a blank URL refuses to install. For a HuggingFace file the hash to match is `lfs.sha256` from `https://huggingface.co/api/models/<repo>?blobs=true`.

## Clustering (dgx-monarch)

[dgx-monarch](https://github.com/Deen-Media/dgx-monarch) splits one render across several Sparks over the ConnectX fabric, or runs one prompt per Spark, from a single ComfyUI window. Upstream measured MiniMax H3 INT8 (1280×720, 241 frames) at 42 min on one Spark and 23 min on two. Here it runs in the same container image as ComfyUI, with the same update and rollback commands, and nothing installed on the host.

**It is off until you create `spark-cluster.toml`.** Without that file every command behaves as above. With it, `install`/`update` build a `cluster` image (the normal image plus a thin dgx-monarch layer), and the containers run on the host network so Monarch and NCCL can reach the fabric. `SPARK_CLUSTER=0` forces stand-alone even with the file present.

> [!NOTE]
> Tested on two Sparks over a RoCE fabric. Larger fleets use the same setup.

### Roles

- **Driver**: the one Spark that runs ComfyUI. You open its UI, and it attaches the workers when a dgx-monarch node runs.
- **Worker**: one per Spark, **the driver's included**. It's a detached container (`<CONTAINER_NAME>-worker`) with a docker restart policy, listening on the Spark's fabric address (port 26600 by default).
- The first `[[hosts]]` entry is rank 0 and the NCCL master. Make it the driver Spark.

### Before you start

On every Spark:

1. **This repo at the same path**, on the same branch, with `install` done once. `cluster push` restarts remote workers with `ssh <host> <that path>/spark-comfyui.sh cluster worker --restart`. If the path differs, set `SPARK_REMOTE_SCRIPT`.
2. **The same instance names.** If you use a `spark-comfyui.env`, use the same file everywhere. `cluster push` passes the image and container names along, but commands you run by hand on a worker read that worker's own file.
3. **The same content.** The same models at the same relative paths under `models/`, and the same custom nodes (the same `comfyui-nodes.list`, plus `recipe install` on each). dgx-monarch loads the model on every rank.
4. **A fabric address.** Each Spark needs an IPv4 address on the ConnectX ports. Check them with `ip -br addr` and `ibdev2netdev`. A switch works for 2 to 64 Sparks with `fabric_profile = "generic-roce"`. Two Sparks on one direct cable can use `dgx-spark-pair`, the profile upstream measured.
5. **A locked-down fabric.** The Monarch attach API has **no peer authentication**, and actors open dynamic ports, so a firewall rule for port 26600 alone isn't enough. Keep the fabric a dedicated network and source-restrict *all* traffic on it to the cluster hosts. Read upstream's [SECURITY.md](https://github.com/Deen-Media/dgx-monarch/blob/main/SECURITY.md#fabric-trust-boundary).

### Step by step

**1. Seed the config and the cluster ssh key** (on the driver):

```bash
./spark-comfyui.sh cluster init
```

This writes `spark-cluster.toml` (gitignored, never overwritten) and creates a cluster key in `~/.ssh/<CONTAINER_NAME>`, for example `~/.ssh/spark-comfyui-cluster`. ComfyUI uses that key to reach the workers. A re-run prints the public key again.

**To reuse a key every Spark already accepts**, copy it into that folder and point `ssh_key` at it:

```bash
mkdir -m 700 -p ~/.ssh/spark-comfyui-cluster
cp ~/.ssh/id_ed25519_shared ~/.ssh/spark-comfyui-cluster/   # your key's name
# spark-cluster.toml: ssh_key = "~/.ssh/id_ed25519_shared"   (the container's ~/.ssh is that folder)
```

Custom nodes running in ComfyUI can read the key in that folder.

**2. Authorize ssh.** It takes two kinds of access:

```bash
# a) the cluster key, on EVERY Spark including the driver itself
#    (the driver probes its own worker by its fabric IP)
ssh-copy-id -i ~/.ssh/spark-comfyui-cluster/id_ed25519.pub you@192.168.100.12
# b) the fabric addresses as known hosts for that key
ssh-keyscan -H 192.168.100.11 192.168.100.12 192.168.100.13 >> ~/.ssh/spark-comfyui-cluster/known_hosts
# c) your OWN passwordless ssh from the driver host to every worker:
#    'cluster push' and 'cluster status' use it, outside any container
ssh 192.168.100.12 true
```

**3. Edit `spark-cluster.toml`.** Replace every `192.0.2.x` placeholder with the fabric IPs. Use one `[[hosts]]` block per Spark, the driver first, and the same `gpus = 1` everywhere. Name your NICs under `[fabric.generic-roce]`. Leave `python`, `auto_heal` and `comfy_dir` as seeded. Set `ssh_user` on every host to your account on that Spark (`cluster init` fills in yours). A three-Spark switch looks like this:

```toml
[cluster]
client_bind        = "tcp://192.168.100.11:0"   # the driver's fabric IP
fabric_profile     = "generic-roce"
transport_security = "trusted_fabric"
nccl_master_addr   = "192.168.100.11"
nccl_master_port   = 29777
python             = "~/.local/share/spark-comfyui-cluster/python"
auto_heal          = false
comfy_dir          = "/opt/ComfyUI"   # ComfyUI inside the containers
ssh_key            = "~/.ssh/id_ed25519"

[[hosts]]
name = "192.168.100.11"
address = "tcp://192.168.100.11:26600"
gpus = 1
ssh_user = "you"

[[hosts]]
name = "192.168.100.12"
address = "tcp://192.168.100.12:26600"
gpus = 1
ssh_user = "you"

[[hosts]]
name = "192.168.100.13"
address = "tcp://192.168.100.13:26600"
gpus = 1
ssh_user = "you"

[fabric.generic-roce]
NCCL_SOCKET_IFNAME = "enp1s0f0np0"
GLOO_SOCKET_IFNAME = "enp1s0f0np0"
NCCL_IB_HCA        = "rocep1s0f0,roceP2p1s0f0"
NCCL_NET_GDR_LEVEL = "SYS"   # GB10 has no GPUDirect RDMA
UCX_NET_DEVICES    = "enp1s0f0np0"
```

Use names that are fabric IPs: that's how each Spark finds its own row, so one file serves every Spark. Fabric NCCL settings go here, not in the environment: the worker passes only an allowlist of variables on to its actors. Full key reference: upstream [CLUSTER.md](https://github.com/Deen-Media/dgx-monarch/blob/main/docs/CLUSTER.md).

**4. Copy the file to every Spark**, next to `spark-comfyui.sh`:

```bash
for h in 192.168.100.12 192.168.100.13; do scp -p spark-cluster.toml "$h":~/Documents/ComfyUI/spark-comfyui-cluster/; done   # same checkout path everywhere
```

**5. Build on the driver, then push:**

```bash
./spark-comfyui.sh update          # cluster image: ComfyUI + dgx-monarch at their pinned commits
./spark-comfyui.sh cluster push    # docker save | ssh docker load to every other [[hosts]] entry
```

Workers never build. dgx-monarch refuses to attach ranks whose sources differ, so every Spark runs the driver's exact image. `push` keeps the remote's old image as `:previous`, compares image IDs, and restarts the remote worker (`--no-restart` skips that). `cluster push 192.168.100.13` sends to one host.

**6. Start the workers.** `push` already restarted the remote ones. On the driver Spark:

```bash
./spark-comfyui.sh cluster worker
```

This finds the `[[hosts]]` row whose IP is local, starts the worker container, and installs a small helper at `~/.local/share/spark-comfyui-cluster/python`. `docker ps` shows the worker `healthy` once it listens. Set `SPARK_WORKER_ADDRESS=tcp://IP:26600` if `hostname -I` doesn't list the fabric IP.

**7. Check, then start ComfyUI:**

```bash
./spark-comfyui.sh cluster status        # worker health + image/monarch commit on every host, then dgxm status
./spark-comfyui.sh cluster dgxm doctor   # dgx-monarch's own end-to-end checks, run in the container
./spark-comfyui.sh service               # or 'run'; UI on http://<driver-ip>:$PORT
```

dgx-monarch's example workflows appear under **Workflow → Browse Templates**. For MiniMax H3, start from `dgx-monarch-minimax-h3-t2va` (or `-mp4`, `-guide`). They replace the stock loader and sampler with `DGXMonarchUNETLoader` and `DGXMonarchKSampler`, behind a `DGXMonarchInit` node that attaches the fleet. Upstream's `example_workflows/artifacts.toml` lists the model files they expect. The first render attaches every worker and loads the model on each rank, so it takes longer than the ones after it. For Continuity there's nothing to change: see [Continuity on the cluster](#continuity-on-the-cluster).

### Quick test: one Spark, then two

A first run of the fork on two Sparks, A (the driver) and B, either of which may also have a stock install running. Phase 1 proves each Spark on its own. Phase 2 clusters them. Commands run in the checkout directory unless a step says otherwise.

#### Phase 1: each Spark on its own

On **both** Sparks:

```bash
# 1. Clone at the SAME path on both ('cluster push' restarts B's worker by that path)
git clone https://github.com/audiohacking/spark-comfyui-cluster.git ~/Documents/ComfyUI/spark-comfyui-cluster
cd ~/Documents/ComfyUI/spark-comfyui-cluster

# 2. Only if a stock spark-comfyui runs on this Spark: own names and port
cat > spark-comfyui.env <<'EOF'
CONTAINER_NAME=spark-comfyui-cluster
CONTAINER_IMAGE=spark-comfyui-cluster
PORT=8182
EOF

# 3. Already have models on this Spark (a stock install's data/)? Share them, don't
#    download again, BEFORE the first run. Details: "Sharing an existing data folder"
echo "models = ~/Documents/ComfyUI/spark-comfyui/data/models" > spark-mounts.conf
#    ...or use that whole data folder, models, nodes and workflows (it must exist):
#    echo "DATA_DIR=$HOME/Documents/ComfyUI/spark-comfyui/data" >> spark-comfyui.env

# 4. Build (stand-alone: no spark-cluster.toml yet) and check
./spark-comfyui.sh install
./spark-comfyui.sh doctor            # every line PASS
./spark-comfyui.sh status            # 'instance:' shows the names and port above

# 5. Run, open http://<spark-ip>:8182, queue a small workflow
./spark-comfyui.sh run
```

Pass criteria: `doctor` is green, the UI loads, and a generation finishes on each Spark. If a `spark-cluster.toml` is already in the checkout, prefix the commands with `SPARK_CLUSTER=0` for this phase. The stock container, if there is one, is still `Up` in `docker ps`. If you plan to time the cluster later, also run the MiniMax H3 workflow you will cluster here, on A alone, and note the time. Stop with Ctrl-C.

#### Phase 2: clustered

**Pre-flight**, from A. Each line must succeed before going on (`192.168.100.10` is A's fabric IP and `192.168.100.11` is B's; use yours):

```bash
ip -br addr | grep 192.168.100.        # A's fabric IP is up
ping -c2 192.168.100.11                 # B answers over the fabric
ssh 192.168.100.11 docker info >/dev/null && echo ok   # passwordless ssh, docker without sudo on B
ssh 192.168.100.11 test -x ~/Documents/ComfyUI/spark-comfyui-cluster/spark-comfyui.sh && echo ok   # same path on B
ss -ltn | grep -E ':(26600|29777|8182)\b' || echo free  # ports free on A; run it on B too
```

`cluster init` seeds `ssh_user` with your account on every host. If your user name differs on B, change B's row in step 2. To reuse a key both Sparks already accept instead of the new one, see [step 1](#step-by-step).

**On A, the driver:**

```bash
# 1. config + cluster key
./spark-comfyui.sh cluster init
ssh-copy-id -i ~/.ssh/spark-comfyui-cluster/id_ed25519.pub 192.168.100.10   # A itself
ssh-copy-id -i ~/.ssh/spark-comfyui-cluster/id_ed25519.pub 192.168.100.11   # B
ssh-keyscan -H 192.168.100.10 192.168.100.11 >> ~/.ssh/spark-comfyui-cluster/known_hosts

# 2. edit spark-cluster.toml: the 192.0.2.x placeholders -> 192.168.100.10 (first) and .11,
#    ssh_user = your account on each Spark (seeded with yours),
#    fabric_profile = "dgx-spark-pair" for one direct cable, or "generic-roce" + NIC names
#    for a switch (see the example above; 'ibdev2netdev' lists the NICs)
${EDITOR:-nano} spark-cluster.toml

# 3. the same file on B
scp spark-cluster.toml 192.168.100.11:~/Documents/ComfyUI/spark-comfyui-cluster/

# 4. build the cluster image, send it to B (this also starts B's worker), start A's worker
./spark-comfyui.sh update
./spark-comfyui.sh cluster push
./spark-comfyui.sh cluster worker
```

**Check before the first render:**

```bash
./spark-comfyui.sh cluster status       # both hosts: worker healthy, SAME image ID and monarch commit, no WARN
./spark-comfyui.sh cluster dgxm doctor  # dgx-monarch's own checks, all green
ssh -i ~/.ssh/spark-comfyui-cluster/id_ed25519 192.168.100.11 '~/.local/share/spark-comfyui-cluster/python -c "print(1)"'   # prints 1
```

B's worker can stay `starting` for a few minutes on its first start (node requirements and GPU checks). `ssh 192.168.100.11 docker logs -f spark-comfyui-cluster-worker` shows its progress.

**Render:**

```bash
./spark-comfyui.sh run                  # UI on http://<A-ip>:8182
```

1. Open **Workflow → Browse Templates** and load `dgx-monarch-minimax-h3-t2va`. Point its loader at your MiniMax H3 model, and keep the frame count small for the first run.
2. Queue it. The first run attaches both workers and loads the model on each rank, so it is slow.
3. In a second terminal on A, run `./spark-comfyui.sh cluster dgxm top`. Both hosts should show GPU load during sampling.
4. Queue it again at the size you timed in phase 1, and compare.
5. For Continuity, load **Browse Templates → spark-continuity-monarch → "Continuity - MiniMax H3 on the DGX cluster"**, or your own Continuity workflow, and queue it. The log names what ran on the cluster and what ran locally.

When something fails, check the [cluster troubleshooting](#cluster-troubleshooting) list first. Note the step and the `cluster status` / `cluster dgxm doctor` output for an issue.

**Back to stand-alone after testing:**

```bash
./spark-comfyui.sh cluster worker --disable   # on A, and on B
mv spark-cluster.toml spark-cluster.toml.off  # on A and B; stand-alone again from the next command
```

The cluster image still runs stand-alone. `update` rebuilds the plain one.

### Continuity on the cluster

In cluster mode, [Continuity](https://github.com/roadmaus/ComfyUI-Continuity)'s MiniMax H3 video renders sample across the Sparks automatically, with no change to the workflow.

- **Start from the template:** **Workflow → Browse Templates → spark-continuity-monarch → "Continuity - MiniMax H3 on the DGX cluster"**, or open any Continuity workflow.
- **What runs where:** the main pass runs on the cluster. These run on the driver Spark:
  - the face, refine, re-detail and seam-restore passes;
  - any shot with a turbo lead-in, a non-default sigma shift, a sound seam, a ControlNet guide or a model-patch accelerator.

  `comfyui_<PORT>.log` lists what ran where.
- **Previews** work as on one GPU. For taeh3 previews, put [`taeh3.safetensors`](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors) in `models/vae_approx`.
- **First render:** the first render of a checkpoint and LoRA combination starts with dgx-monarch's identity gate, two short proof renders. Later renders skip it.
- **Memory:** a render with a local pass loads a second copy of the model on the driver Spark, about 21 GB for H3 int8. Keep that much free there.
- **Opting out:** `SPARK_CONTINUITY_MONARCH=0` in `spark-comfyui.env` keeps Continuity renders on the driver's GPU.
- **Status:** `curl http://<driver>:<PORT>/spark/continuity-monarch`.

### Day to day

| Task | Command |
|---|---|
| Upgrade ComfyUI and dgx-monarch | `update`, then `cluster push` |
| Roll back both | `update --rollback`, then `cluster push` |
| Keep a known-good pair | `update --keep=h3-ok` pins the image you leave as `:keep-h3-ok` |
| Follow dgx-monarch's main branch | `MONARCH_REF=main ./spark-comfyui.sh update`, then `cluster push` |
| Pin an exact dgx-monarch | `MONARCH_REF=v1.3.0` (any tag, branch or commit) |
| Add a Spark | Add a `[[hosts]]` block, copy the file everywhere, `cluster push <ip>`, then restart the driver |
| Restart a worker | `cluster worker --restart` on that Spark |
| Remove a worker | `cluster worker --disable` |
| Watch the fleet | `cluster dgxm top` |
| Back to stand-alone | `cluster worker --disable` on each Spark, move `spark-cluster.toml` aside, `update` (or set `SPARK_CLUSTER=0`) |

`doctor` says when a newer dgx-monarch release exists. Its commit is stamped on the image (`org.spark-comfyui.monarch-sha`), so `:previous` and `:keep-*` snapshot it together with ComfyUI.

### Container-mode differences from upstream dgx-monarch

- `dgxm up`, `down`, `restart`, `setup`, `init`, `update`, `install-service`, `uninstall` and `launcher` manage bare-metal systemd services. `cluster dgxm` refuses them and names the equivalent: `cluster worker`, `update` + `cluster push`.
- Healing is docker's restart policy, so `auto_heal = false`.
- The node pack loads from the image (`/opt/monarch-nodes`), not from `custom_nodes/`. Don't also install dgx-monarch through Manager or `comfyui-nodes.list`.
- Containers use the host network, so ComfyUI listens on every interface unless `BIND_ADDR` is set.

### Cluster troubleshooting

- **Attach refused: listener not found / identity mismatch**: run `ssh -i ~/.ssh/spark-comfyui-cluster/id_ed25519 <user>@<ip> ~/.local/share/spark-comfyui-cluster/python -c 'print(1)'` from the driver host. It must print `1`. If it doesn't, check the key, `known_hosts`, `ssh_user`, and that the worker is running.
- **Attach refused: source mismatch**: the hosts run different images. Run `cluster push`.
- **NCCL hangs or picks the wrong NIC**: set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` under `[fabric.generic-roce]`, then run `cluster worker --restart` everywhere. Add `NCCL_DEBUG = "INFO"` there to see what NCCL chose.
- **A worker stays `starting`**: on first start it installs custom-node requirements before it listens, which can take a few minutes. Follow it with `docker logs -f <CONTAINER_NAME>-worker`.
- **`cluster status` prints WARN lines**: fix the named settings in `spark-cluster.toml` before attaching.
- **The mesh shows `poisoned`**: restart ComfyUI with `docker restart <CONTAINER_NAME>`.
- **`dgxm doctor` warns "mesh health: a local driver gave no readable answer"** while another ComfyUI runs on the driver Spark: harmless.
- **`[dgxm:C guard=loader_footprint_preflight …]`**: click **Load with slab residency** on the model's card in the DGX Monarch panel, or set `DGXM_ALLOW_SLAB_RESCUE=1`, and queue again.
- **`[dgxm:C guard=activation_footprint_preflight …]`**: the render doesn't fit in memory. Use smaller or shorter reference videos (they often dominate), a smaller canvas or a shorter clip, or free memory on the Sparks, for example by stopping other ComfyUI instances.

### Settings

| Variable | Default | What it does |
|---|---|---|
| `CLUSTER_CONF` | `spark-cluster.toml` | The cluster switch and dgx-monarch's `cluster.toml`. |
| `SPARK_CLUSTER` | unset | `0` forces stand-alone even with the file present. |
| `MONARCH_REF` | newest `vX.Y.Z` | dgx-monarch tag, branch or commit to build. |
| `MONARCH_REPO` | upstream | dgx-monarch git URL, for a fork. |
| `CLUSTER_SSH_DIR` | `~/.ssh/$CONTAINER_NAME` | The driver container's ssh identity. |
| `WORKER_NAME` | `<CONTAINER_NAME>-worker` | This Spark's worker container. |
| `SPARK_WORKER_ADDRESS` | from the config | This Spark's worker bind, when `hostname -I` does not list its fabric IP. |
| `SPARK_REMOTE_SCRIPT` | this path | Where `spark-comfyui.sh` lives on the other Sparks. |
| `SPARK_CONTINUITY_MONARCH` | `1` | `0` keeps Continuity's H3 renders on the driver's card in cluster mode. |
| `SPARK_MONARCH_SLAB` | unset | Weight residency for Continuity cluster renders. Unset uses dgx-monarch's default; `on` maps the model from the file (less memory per Spark, but its first-use check needs extra memory on large renders). |
| `SPARK_MONARCH_PREVIEW_FRAMES` | `17` | Frames in the live preview of a Continuity cluster render, spread across the clip. `0` previews every frame, which costs several seconds of GPU per step. |
| `SPARK_MONARCH_UNLOAD_AFTER` | `1` | After each Continuity cluster render, the driver's ComfyUI unloads its models so the next render has the memory. `0` keeps them loaded. |
| `SPARK_MONARCH_RELEASE_ON_SWITCH` | `1` | When a Continuity render asks for a different diffusion model than the cluster holds, the workers are reset first (about a minute) so the previous model's memory is free. `0` keeps it loaded. |
| `SPARK_MONARCH_NOTICES` | essentials | dgx-monarch's pop-up notices in ComfyUI: only warnings and errors by default. `all` shows every notice; all of them stay in the DGX Monarch panel either way. |

## Troubleshooting

Start with `./spark-comfyui.sh doctor`; every failure names its fix. Common ones:

- **An update broke generation**: `update --rollback`, restart.
- **A custom node will not load**: check the start log; the entrypoint installs each node's requirements and warns per node. It uses `uv` when the venv has it and falls back to `pip`, so a node with pins too sloppy for uv's resolver still installs. A node needing a system library the image lacks is worth an issue.
- **The container is up but the UI does not load**: `doctor`. A hung server still looks `Up` to docker, whose restart policy only reacts to an exit, so the image carries a health check against `/system_stats`. `doctor` turns that into a pass or a fail, and `status` says `NOT ANSWERING` next to the process line.
- **Silent hard-reboot during video generation**: `status --watch`, reproduce, read the last logged lines. A power spike right before death is overcurrent; fix with `tune --clock-cap 2100`.
- **`run` fails with permission errors on your own content**: the container runs as whoever launched it, so this means the files belong to someone else. An image built before 2026.08.23 baked uid 1000 and wrote as that user no matter who you were, which broke every host whose account was not uid 1000. Update, then clear the residue once with `sudo chown -R "$(id -u):$(id -g)" data/`. `doctor` names the directory and the owning uid.
- **Machine freezes near memory limit**: swap thrash on unified memory; run `tune`.
- **Docker is eating the disk**: `prune`. Images and BuildKit cache are the two things that grow silently, and a full filesystem stops a rebuild dead. It keeps `:latest`, `:previous` and every `keep-*` pin, shows exactly what goes first, and never rebuilds. `doctor` reports both numbers on every run.

  ```console
  $ ./spark-comfyui.sh prune --yes

    Leftover image tags (nothing references these):
      spark-comfyui:2026.07.26                       11.3GB
      spark-comfyui:2026.07.27                       11.3GB
      spark-comfyui:2026.07.28                       11.3GB
      spark-comfyui:2026.07.29                       11.4GB
      spark-comfyui:2026.07.31                       11.4GB
      spark-comfyui:2026.08.02                       11.4GB
    Untagged spark-comfyui images (matched by image label):
      906dbbaf56d4                                   11.3GB
    Build cache: 95 GB reclaimable, trimming to 7d / 40 GB

    Kept: spark-comfyui:latest, :previous, and every keep-* pin.
    Your content (data/) and the run-time cache volume are not touched.

  ==> Removed 7 leftover image(s)
    Disk now: 273G free of 917G (69% used)
  ```
- **LTX-2.3 audio workflows fail with `Input type (float) and bias type (c10::BFloat16)`**: the audio VAE never casts the input waveform to the VAE dtype, so it cannot run under `--bf16-vae`. This affects every LTX-2.3 audio workflow, including the templates ComfyUI ships. Start with `--no-bf16-vae`; the unet and text-encoder speedups are kept. You cannot fix this by passing `--fp32-vae` instead, because ComfyUI's VAE precision flags are mutually exclusive and the entrypoint has already added `--bf16-vae`.
- **Not sure SageAttention is really active?**: on this build it cannot silently fall back to plain attention, the failure mode most Spark guides warn about. The image compiles a native sm_121 kernel, and every start runs a live multi-shape kernel test that refuses to launch if it fails. `doctor` runs the same gate on demand.

> [!NOTE]
> The `sm_121 exceeds torch's supported maximum` line at startup is expected on GB10 and harmless. PyTorch JITs the kernels through PTX.

## Security notes

- Custom nodes run as a non-root user with all capabilities dropped and only your content directories and the GPU visible. A malicious node cannot read your SSH keys or anything else on the host.
- The image is reproducible from this repo: both base images pinned by digest as well as tag, pinned ComfyUI commit, pinned SageAttention commit, sha256-pinned onnxruntime wheel from PyPI. `update --rollback` returns to the previous image atomically.
- **`install` and `update` seed `comfyui-nodes.list` with one active entry**, so a fresh install, and an upgrade of an existing one, downloads and runs one third-party node from the Comfy Registry ([comfyui-workflow-models-downloader](https://registry.comfy.org/publishers/slahiri/nodes/comfyui-workflow-models-downloader)). Both print the file when they write it. Comment the line out before your next `run` if you want an install that executes no third-party node code. Everything the containment above says still applies to it.
- **Cluster mode widens the container**, and only when `spark-cluster.toml` exists. It shares the host's network and process table (`--network host --pid host`) and gets the RDMA devices. The driver container also gets a cluster-only ssh key (never your `~/.ssh`), which the custom nodes in it can read. The other limits stay: non-root, all capabilities dropped, `no-new-privileges`. The Monarch attach API has **no peer authentication**, so the fabric must be dedicated and source-restricted to the cluster hosts on every port. Read upstream's [SECURITY.md](https://github.com/Deen-Media/dgx-monarch/blob/main/SECURITY.md#fabric-trust-boundary) before enabling it. With host networking ComfyUI listens on every interface unless `BIND_ADDR` is set.
- The UI is published on every interface by default, so anything on your LAN can reach it. `BIND_ADDR=127.0.0.1` restricts it to the box itself; reach it from elsewhere over an SSH tunnel.

> [!WARNING]
> ComfyUI has no authentication. Manager's `personal_cloud` mode is fine on a trusted LAN. Do not expose the port to the internet.

---

MIT, see [LICENSE](LICENSE). The GB10 knowledge here comes from the NVIDIA DGX Spark developer forums, the [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks), and the community projects that mapped this hardware in public.
