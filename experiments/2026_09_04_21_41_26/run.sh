#!/usr/bin/env bash
set -euo pipefail
cd /home/mengh/RadonBridge
export PYTHONPATH=.:third_party/MHD_Project
export TORCH_HOME=/data/mengh/RadonBridge/weights
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python scripts/run_ablation_experiment.py --phase run --output /data/mengh/RadonBridge/runs/2026_09_04_21_41_26 --protocol experiments/2026_09_04_21_41_26/protocol.json --data /data/mengh/RadonBridge/cache/full1264_296
