#!/usr/bin/env bash
# Per-port RDMA traffic and the management/default-route NIC over 10 s, on one
# Spark (run on each during a render). EXTRA_IFS adds interfaces, e.g. a VPN's.
ifs="$(ip route show default | awk '{print $5}' | head -1) ${EXTRA_IFS:-}"
snap(){ for d in rocep1s0f0 roceP2p1s0f0 rocep1s0f1 roceP2p1s0f1; do echo "$d $(cat /sys/class/infiniband/$d/ports/1/counters/port_xmit_data) $(cat /sys/class/infiniband/$d/ports/1/counters/port_rcv_data)"; done
        for n in $ifs; do echo "$n $(cat /sys/class/net/$n/statistics/tx_bytes) $(cat /sys/class/net/$n/statistics/rx_bytes)"; done; }
a=$(snap); sleep 10; b=$(snap)
echo "$(hostname) over 10s (tx/rx Gb/s):"
paste <(echo "$a") <(echo "$b") | awk '{ m=($1 ~ /^roce/)?4:1; printf "  %-13s tx %6.2f  rx %6.2f\n", $1, ($5-$2)*m*8/10/1e9, ($6-$3)*m*8/10/1e9 }'
