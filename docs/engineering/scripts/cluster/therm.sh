echo "== $(hostname)"
echo "cmdline: $(cat /proc/cmdline | tr ' ' '\n' | grep -vE '^(BOOT_IMAGE|root|ro|quiet|splash)' | tr '\n' ' ')"
echo "clocks: $(nvidia-smi --query-gpu=clocks.applications.graphics,clocks.max.sm,power.limit,power.max_limit,enforced.power.limit --format=csv,noheader 2>&1)"
nvidia-smi -q -d CLOCK 2>/dev/null | grep -A3 -iE "locked|Clock Policy|Applications Clocks" | head -12
echo "modprobe.d: $(ls /etc/modprobe.d | tr '\n' ' ')"; grep -rhv '^#' /etc/modprobe.d/*nvidia* 2>/dev/null | grep . 
echo "services: $(systemctl list-unit-files --state=enabled --no-pager | grep -iE 'gb10|clock|therm|power|fan|tune|nvidia' | awk '{print $1}' | tr '\n' ' ')"
for u in gb10-clock-cap.service; do systemctl cat $u 2>/dev/null | grep -vE '^\s*$'; done
echo "firmware: $(cat /sys/class/dmi/id/bios_version 2>/dev/null) ec=$(cat /sys/class/dmi/id/ec_firmware_release 2>/dev/null)"
dpkg -l 2>/dev/null | grep -iE 'thermal|gb10|dgx-spark|nvidia-spark|fwupd' | awk '{print $2"="$3}' | tr '\n' ' '; echo
