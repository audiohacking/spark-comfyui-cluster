#!/usr/bin/env bash
# Run a2a_bench.py on every Spark in spark-cluster.toml at once, in their
# worker containers, with the cluster's own fabric settings.
#   a2a_bench.sh [VAR=VALUE ...]   extra NCCL/A2A env on top of the fabric section
# Run on the driver Spark as the cluster user, with no render running.
# CLUSTER_CONF overrides the config path (default: the repo's spark-cluster.toml).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
CONF=${CLUSTER_CONF:-$REPO/spark-cluster.toml}
mapfile -t HOSTS < <(python3 "$REPO/lib/cluster_read.py" "$CONF" hosts | cut -f1)
[[ ${#HOSTS[@]} -ge 2 ]] || { echo "need at least two [[hosts]] in $CONF" >&2; exit 2; }
# [fabric.<fabric_profile>] as KEY=VALUE, the way the workers apply it
mapfile -t BASE < <(python3 - "$CONF" <<'PY'
import sys, tomllib
cfg = tomllib.load(open(sys.argv[1], "rb"))
profile = cfg.get("cluster", {}).get("fabric_profile", "generic-roce")
for key, value in (cfg.get("fabric", {}).get(profile) or {}).items():
    print(f"{key}={value}")
PY
)
ENV=("${BASE[@]}" "$@" MASTER_ADDR=${HOSTS[0]} MASTER_PORT=${A2A_PORT:-29888} WORLD_SIZE=${#HOSTS[@]})
here() { ip -o -4 addr show | grep -qw "inet $1"; }
pids=()
for i in "${!HOSTS[@]}"; do
  h=${HOSTS[$i]}
  args=(); for kv in "${ENV[@]}" "RANK=$i"; do args+=(-e "$kv"); done
  cmd="docker exec -i ${args[*]} spark-comfyui-cluster-worker timeout 300 python -"
  if here "$h"; then bash -c "$cmd" < "$HERE/a2a_bench.py" > "/tmp/a2a.$i.log" 2>&1 &
  else ssh -o BatchMode=yes "$h" "$cmd" < "$HERE/a2a_bench.py" > "/tmp/a2a.$i.log" 2>&1 & fi
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -hE "^(a2a|p2p|bidir|ring) |Error|error" /tmp/a2a.0.log | tail -3
