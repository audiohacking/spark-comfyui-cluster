snap(){ for d in rocep1s0f0 roceP2p1s0f0 rocep1s0f1 roceP2p1s0f1; do echo "$d $(cat /sys/class/infiniband/$d/ports/1/counters/port_xmit_data) $(cat /sys/class/infiniband/$d/ports/1/counters/port_rcv_data)"; done; echo "enP7s7 $(cat /sys/class/net/enP7s7/statistics/tx_bytes) $(cat /sys/class/net/enP7s7/statistics/rx_bytes)"; echo "vpn0 $(cat /sys/class/net/vpn0/statistics/tx_bytes) $(cat /sys/class/net/vpn0/statistics/rx_bytes)"; }
a=$(snap); sleep 10; b=$(snap)
echo "$(hostname) over 10s (tx/rx Gb/s):"
paste <(echo "$a") <(echo "$b") | awk '{ m=($1 ~ /^roce/)?4:1; printf "  %-13s tx %6.2f  rx %6.2f\n", $1, ($5-$2)*m*8/10/1e9, ($6-$3)*m*8/10/1e9 }'
