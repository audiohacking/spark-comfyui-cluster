bad=0; out=""
for d in FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP; do
  ip -4 a | grep -q "inet $d/" && continue
  dev=$(ip route get $d | grep -oE 'dev \S+' | cut -d' ' -f2); ping -c1 -W1 $d >/dev/null && p=ok || { p=FAIL; bad=1; }
  case $dev in enp1*|enP2*) ;; *) bad=1;; esac; out="$out $d:$dev:$p"
done
echo "$(hostname): $([ $bad = 0 ] && echo ALL-FABRIC-OK || echo PROBLEM)"; echo " $out" | sed "s/ /\n  /g"
