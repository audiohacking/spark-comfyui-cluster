#!/usr/bin/env bash
# Seed a new Spark over the fabric from this one: both repos, data (models in
# parallel streams), then report what still differs.
#   sync_spark.sh DEST_FABRIC_ADDR
set -u
D=${1:?usage: $0 DEST_FABRIC_ADDR}; B=${SPARK_BASE:-$HOME/Documents/ComfyUI}
E="ssh -o BatchMode=yes -c aes128-gcm@openssh.com"
ip route get "$D" | grep -qE 'dev (enp1|enP2)' || { echo "route to $D is not on the fabric"; exit 1; }
ssh -o BatchMode=yes "$D" "mkdir -p $B"
t0=$(date +%s)
rsync -aH -e "$E" --exclude 'data/models/' "$B/spark-comfyui-cluster" "$B/spark-comfyui" "$D:$B/" && echo "repos+data(no models) done $(( $(date +%s)-t0 ))s"
ssh -o BatchMode=yes "$D" "mkdir -p $B/spark-comfyui/data/models"
cd "$B/spark-comfyui/data/models" || exit 1
ls -A | xargs -P 6 -I{} rsync -aH -e "$E" "{}" "$D:$B/spark-comfyui/data/models/"
echo "models rc=$? done $(( $(date +%s)-t0 ))s"
rsync -aHn --itemize-changes -e "$E" "$B/spark-comfyui/data/models/" "$D:$B/spark-comfyui/data/models/" | wc -l | sed 's/^/models differences remaining: /'
