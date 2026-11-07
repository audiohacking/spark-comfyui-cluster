#!/usr/bin/env bash
# Ask every Spark's Workers to profile one sampling step of the next render,
# or collect the results. Run on the driver Spark as the cluster user.
#   step_profile.sh request ID [STEP]   # STEP is 0-based, default 3
#   step_profile.sh collect ID DEST     # copies every rank's files to DEST
# The request lives in each host's dgx-monarch cache dir, which the worker
# container mounts at ~/.cache/dgx-monarch (actor/step_profile.py, patch
# monarch-step-profile). Each id is served once per Worker process.
# Hosts come from spark-cluster.toml (CLUSTER_CONF overrides the path).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
CONF=${CLUSTER_CONF:-$REPO/spark-cluster.toml}
mapfile -t HOSTS < <(python3 "$REPO/lib/cluster_read.py" "$CONF" hosts | cut -f1)
# The host side of the worker's ~/.cache/dgx-monarch (spark-comfyui.sh's default state dir).
DIR=.local/state/spark-comfyui-cluster/.cache/dgx-monarch
# This Spark is reached directly: ssh to its own fabric address may have no known host key.
here() { ip -o -4 addr show | grep -qw "inet $1"; }
on() { local h=$1; shift; if here "$h"; then bash -c "$*"; else ssh -o BatchMode=yes "$h" "$*"; fi; }
cmd=${1:?request|collect}; id=${2:?id}
case "$cmd" in
  request)
    step=${3:-3}
    for h in "${HOSTS[@]}"; do
      on "$h" "printf '{\"id\": \"%s\", \"step\": %d}\n' '$id' $step > ~/$DIR/step-profile.request && echo \"$h: requested $id step $step\""
    done ;;
  collect)
    dest=${3:?dest}; mkdir -p "$dest"
    for h in "${HOSTS[@]}"; do
      if here "$h"; then cp ~/"$DIR"/step-profile/"$id"-* "$dest/"; else
        scp -o BatchMode=yes -q "$h:$DIR/step-profile/$id-*" "$dest/"; fi && echo "$h: collected"
    done ;;
  *) echo "usage: $0 request ID [STEP] | collect ID DEST" >&2; exit 2 ;;
esac
