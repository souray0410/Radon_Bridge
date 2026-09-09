#!/bin/bash
set -euo pipefail
umask 077
module load python/3.11.0
unset PYTHONPATH
RB_ENV=/ibex/project/c2377/souray/home/mengh/environments/radon_bridge_2026_09_09_10_30_34
RB_CODE=/ibex/project/c2377/souray/home/mengh/Radon_Bridge/2026_09_09_10_30_34
RB_RUN=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/runs/2026_09_09_10_30_34
export PIP_CACHE_DIR=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/package_cache
for attempt in $(seq 1 360); do
  [ -f "$RB_ENV/dependencies_ready" ] && break
  sleep 60
done
test -f "$RB_ENV/dependencies_ready"
# Recheck under a clean PYTHONPATH to avoid relying on module-system site packages.
"$RB_ENV/bin/python" -m pip install -r "$RB_CODE/experiments/2026_09_09_10_30_34/requirements_ibex.txt"
"$RB_ENV/bin/python" -m pip install --no-deps -e "$RB_CODE"
"$RB_ENV/bin/python" -m pip check > "$RB_RUN/environment_dependency_check.txt"
"$RB_ENV/bin/python" -m pip freeze > "$RB_RUN/environment_versions.txt"
"$RB_ENV/bin/python" -m radon_bridge --workspace "$RB_CODE/experiments/2026_09_09_10_30_34/ibex_workspace.json" --check-framework > "$RB_RUN/environment_preflight.json"
"$RB_ENV/bin/python" -c 'import radon_bridge.graph,radonbridge.graph; assert radon_bridge.graph is radonbridge.graph; print("canonical and legacy import identity passed")' > "$RB_RUN/environment_import_check.txt"
touch "$RB_ENV/environment_accepted"
