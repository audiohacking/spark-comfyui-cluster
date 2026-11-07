#!/usr/bin/env bash
# On one Spark: every given fabric address answers, and through a fabric port
# (enp1s0f*/enP2p1s0f*), never the management NIC or a VPN interface.
#   rt.sh ADDR [ADDR ...]   e.g. every port-half address of every Spark in the ring
[[ $# -gt 0 ]] || { echo "usage: $0 ADDR [ADDR ...]" >&2; exit 2; }
bad=0; out=""
for d in "$@"; do
  ip -4 a | grep -q "inet $d/" && continue
  dev=$(ip route get "$d" | grep -oE 'dev \S+' | cut -d' ' -f2); ping -c1 -W1 "$d" >/dev/null && p=ok || { p=FAIL; bad=1; }
  case $dev in enp1*|enP2*) ;; *) bad=1;; esac; out="$out $d:$dev:$p"
done
echo "$(hostname): $([ $bad = 0 ] && echo ALL-FABRIC-OK || echo PROBLEM)"; echo " $out" | sed "s/ /\n  /g"
