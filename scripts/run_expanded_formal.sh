#!/usr/bin/env bash
set -u
cd /home/mengh/RadonBridge || exit 90
exec 9>/data/mengh/RadonBridge/.exp008_formal_launch.lock
flock -n 9 || exit 92
run_root=/data/mengh/RadonBridge/runs/exp008_formal
if test -e "$run_root/status.json"; then exit 91; fi
mkdir -p "$run_root"
export CUDA_VISIBLE_DEVICES=1 TORCH_HOME=/data/mengh/RadonBridge/weights
export PYTHONPATH=.:third_party/MHD_Project
/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python -m radonbridge.sweep \
  --data /data/mengh/RadonBridge/cache/full1264_296 --output "$run_root" \
  --protocol experiments/008-expanded-validation/formal_protocol.json \
  --lock /data/mengh/RadonBridge/.active.lock > "$run_root/runner.log" 2>&1
run_status=$?
printf '%s\n' "$run_status" > "$run_root/exit.status"
exit "$run_status"
