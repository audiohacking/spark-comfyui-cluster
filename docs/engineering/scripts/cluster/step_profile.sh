#!/usr/bin/env bash
# Ask every Spark's Workers to profile one sampling step of the next render,
# or collect the results. Run on the driver Spark as the cluster user.
#   step_profile.sh request ID [STEP]   # STEP is 0-based, default 3
#   step_profile.sh collect ID DEST     # copies every rank's files to DEST
# The request lives in each host's dgx-monarch cache dir, which the worker
# container mounts at ~/.cache/dgx-monarch (actor/step_profile.py, patch
# monarch-step-profile). Each id is served once per Worker process.
set -euo pipefail
HOSTS=(FABRIC_IP FABRIC_IP FABRIC_IP)
DIR=.local/state/spark-comfyui-cluster/.cache/dgx-monarch
cmd=${1:?request|collect}; id=${2:?id}
case "$cmd" in
  request)
    step=${3:-3}
    for h in "${HOSTS[@]}"; do
      ssh -o BatchMode=yes "$h" "printf '{\"id\": \"%s\", \"step\": %d}\n' '$id' $step > ~/$DIR/step-profile.request && echo \"$h: requested $id step $step\""
    done ;;
  collect)
    dest=${3:?dest}; mkdir -p "$dest"
    for h in "${HOSTS[@]}"; do
      scp -o BatchMode=yes -q "$h:$DIR/step-profile/$id-*" "$dest/" && echo "$h: collected"
    done ;;
  *) echo "usage: $0 request ID [STEP] | collect ID DEST" >&2; exit 2 ;;
esac
