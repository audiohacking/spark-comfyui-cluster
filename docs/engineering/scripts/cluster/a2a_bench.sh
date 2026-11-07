#!/usr/bin/env bash
# Run a2a_bench.py on the three Sparks at once, in their worker containers.
#   a2a_bench.sh [VAR=VALUE ...]   extra NCCL/A2A env on top of the cluster's fabric env
# Run on the driver Spark as the cluster user, with no render running.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOSTS=(FABRIC_IP FABRIC_IP FABRIC_IP)
BASE=(NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1
      NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
      NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_MERGE_NICS=0 NCCL_NET_GDR_LEVEL=SYS NCCL_IB_GID_INDEX=-1)
ENV=("${BASE[@]}" "$@" MASTER_ADDR=FABRIC_IP MASTER_PORT=29888 WORLD_SIZE=${#HOSTS[@]})
here() { ip -o -4 addr show | grep -qw "inet $1"; }
pids=()
for i in "${!HOSTS[@]}"; do
  h=${HOSTS[$i]}
  args=(); for kv in "${ENV[@]}" "RANK=$i"; do args+=(-e "$kv"); done
  cmd="docker exec -i ${args[*]} spark-comfyui-cluster-worker timeout 300 python -"
  if here "$h"; then bash -c "$cmd" < "$HERE/a2a_bench.py" > /tmp/a2a.$i.log 2>&1 &
  else ssh -o BatchMode=yes "$h" "$cmd" < "$HERE/a2a_bench.py" > /tmp/a2a.$i.log 2>&1 & fi
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -hE "^(a2a|p2p|bidir|ring) |Error|error" /tmp/a2a.0.log | tail -3
grep -h "NCCL INFO.*\(via\|Channel 00\|NET/\|nChannels\|comm \)" /tmp/a2a.0.log | head -${A2A_INFO_LINES:-0}
