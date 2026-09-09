#!/bin/bash
set -euo pipefail
umask 077
module load python/3.11.0
RB_ENV=/ibex/project/c2377/souray/home/mengh/environments/radon_bridge_2026_09_09_10_30_34
export PIP_CACHE_DIR=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/package_cache
if [ ! -x "$RB_ENV/bin/python" ]; then python3 -m venv "$RB_ENV"; fi
"$RB_ENV/bin/python" -m pip install --upgrade pip
"$RB_ENV/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 numpy pandas Pillow scikit-learn scipy
"$RB_ENV/bin/python" -m pip freeze > "$RB_ENV/installed_versions.txt"
"$RB_ENV/bin/python" -c 'import torch,torchvision;print(torch.__version__,torchvision.__version__)'
touch "$RB_ENV/dependencies_ready"
