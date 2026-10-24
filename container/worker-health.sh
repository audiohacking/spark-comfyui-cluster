#!/usr/bin/env bash
# Health check for the dgx-monarch worker role: is something LISTENING on the
# worker address? Read from /proc/net/tcp on purpose. dgx-monarch's own probe
# never opens the worker's protocol socket (a connect is a client to the
# loop), and neither does this.
set -euo pipefail
addr="${SPARK_WORKER_ADDRESS:?}"
hostport="${addr#tcp://}"
ip="${hostport%:*}" port="${hostport##*:}"
# /proc/net/tcp spells the local address as little-endian hex IPv4 + hex port.
IFS=. read -r a b c d <<<"$ip"
want="$(printf '%02X%02X%02X%02X:%04X' "$d" "$c" "$b" "$a" "$port")"
awk -v w="$want" 'NR>1 && $2==w && $4=="0A" {found=1} END {exit !found}' /proc/net/tcp
