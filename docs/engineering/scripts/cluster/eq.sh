#!/usr/bin/env bash
# One host's baseline, for side-by-side comparison across Sparks.
B=${SPARK_BASE:-$HOME/Documents/ComfyUI}   # where spark-comfyui-cluster and spark-comfyui live
echo "kernel=$(uname -r)"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader) pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader) clocks=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader)"
echo "dgx_os=$(grep -h VERSION= /etc/dgx-release 2>/dev/null | tail -1)"
echo "docker=$(docker version --format '{{.Server.Version}}') store=$(docker info --format '{{.Driver}}') ctk=$(nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}')"
echo "swap=[$(swapon --show=NAME --noheadings)] mem=$(free -g | awk '/Mem/{print $2}')G"
echo "services: tune=$(systemctl is-enabled comfyui-tune 2>&1) clockcap=$(systemctl is-enabled gb10-clock-cap 2>&1)/$(systemctl is-active gb10-clock-cap 2>&1)"
echo "units=$(ls /etc/systemd/system/*.service 2>/dev/null | xargs -n1 basename | grep -vE '^(dbus-|display-manager|snap)' | sort | tr '\n' ' ')"
echo "sysctl: $(sysctl -n vm.swappiness net.core.rmem_max net.core.wmem_max net.ipv4.conf.all.rp_filter | tr '\n' ' ')"
echo "memlock=$(ulimit -l) mtu=$(cat /sys/class/net/enp1s0f1np1/mtu) rails=$(rdma link | grep -c ACTIVE)"
echo "repo=$(git -C $B/spark-comfyui-cluster log -1 --format=%h) toml=$(md5sum < $B/spark-comfyui-cluster/spark-cluster.toml | cut -c1-8) env=$(md5sum < $B/spark-comfyui-cluster/spark-comfyui.env | cut -c1-8) mounts=$(md5sum < $B/spark-comfyui-cluster/spark-mounts.conf | cut -c1-8)"
echo "image=$(docker image inspect spark-comfyui-cluster:latest --format '{{.Id}}' | cut -c8-19) worker=$(docker inspect -f '{{.State.Health.Status}} {{.HostConfig.RestartPolicy.Name}}' spark-comfyui-cluster-worker)"
echo "custom_nodes=$(ls $B/spark-comfyui/data/custom_nodes | md5sum | cut -c1-8) ($(ls $B/spark-comfyui/data/custom_nodes | wc -l)) models=$(cd $B/spark-comfyui/data/models && find . -type f -printf '%P %s\n' | sort | md5sum | cut -c1-8)"
echo "shim=$(head -c 200 ~/.local/share/spark-comfyui-cluster/python 2>/dev/null | tail -1)"
