#!/usr/bin/env bash
# Read-only inventory of every node in one allocation. Run on the Slurm login node:
#   bash discover_nodes.sh <jobid>
set -euo pipefail
JOB=$1
N=$(scontrol show job -o "$JOB" | tr " " "\n" | grep "^NumNodes=" | cut -d= -f2)
IMAGE=${IMAGE_REF:?set IMAGE_REF to the pinned image reference by digest}
srun --jobid="$JOB" --overlap --immediate=60 --nodes="$N" --ntasks="$N" --ntasks-per-node=1 --cpu-bind=none --label bash -c '
set -u
echo "host=$(hostname -s) arch=$(uname -m) uid=$(id -u) gid=$(id -g)"
ip -j -4 addr | python3 -c "import json,sys; print(\"addrs=\" + \" \".join(i[\"ifname\"]+\"=\"+a[\"local\"] for i in json.load(sys.stdin) for a in i.get(\"addr_info\",[]) if i[\"ifname\"]!=\"lo\"))"
echo "getent=$(getent hosts "$(hostname -s)" | awk "{print \$1}" | paste -sd,)"
echo "gpus=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | paste -sd";")"
echo "gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "imex=$(ls -ln /dev/nvidia-caps-imex-channels/ 2>/dev/null | awk "NR>1{print \$3\":\"\$4\":\"\$1\":\"\$NF}" | paste -sd,)"
echo "raid=$(df -B1 --output=avail,fstype,target /raid 2>/dev/null | tail -1 | tr -s " ")"
echo "image=$(docker image inspect --format "{{.Id}} {{.Architecture}}" "'"$IMAGE"'" 2>/dev/null || echo MISSING)"
echo "containers=$(docker ps -a --format "{{.Names}}:{{.Status}}" | paste -sd, )"
echo "model_source=$(test -n "${MODEL_SOURCE:-}" && test -f "${MODEL_SOURCE:-}"/config.json && echo present || echo absent)"
for d in /raid/dldata/*/inputs/GLM-5.3-Flash; do [ -d "$d" ] && echo "cache=$d files=$(ls "$d" | wc -l) uid=$(stat -c %u "$d") cfg=$(sha256sum "$d/config.json" | cut -c1-12)"; done
echo "slurm_job=$SLURM_JOB_ID nodelist=$SLURM_JOB_NODELIST"
'
