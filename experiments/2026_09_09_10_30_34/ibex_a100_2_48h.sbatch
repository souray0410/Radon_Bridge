#!/bin/bash
#SBATCH --job-name=Radon_Bridge
#SBATCH --account=pi-mengy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/runs/2026_09_09_10_30_34/logs/%x_%j.out
#SBATCH --error=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/runs/2026_09_09_10_30_34/logs/%x_%j.err
set -euo pipefail
# Template only. Submission remains paused until the user specifies the workload.
RB_CODE_ROOT=/ibex/project/c2377/souray/home/mengh/Radon_Bridge/2026_09_09_10_30_34
RB_RUN_ROOT=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/runs/2026_09_09_10_30_34
RB_ENV=/ibex/project/c2377/souray/home/mengh/environments/radon_bridge_2026_09_09_10_30_34
: "${SLURM_JOB_ID:?Run through Slurm}"
# Workload and its approval are created only after the new protocol is specified.
test -f "$RB_RUN_ROOT/workload.sh" || { echo 'No approved workload; exiting.' >&2; exit 64; }
test -f "$RB_RUN_ROOT/workload.approved" || { echo 'Workload approval missing; exiting.' >&2; exit 64; }
module load python/3.11.0
source "$RB_ENV/bin/activate"
export PYTHONPATH="$RB_CODE_ROOT:$RB_CODE_ROOT/third_party/MHD_Project${PYTHONPATH:+:$PYTHONPATH}"
# Preserve CUDA_VISIBLE_DEVICES as assigned by Slurm; do not use ws02 physical IDs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$RB_CODE_ROOT"
exec srun --kill-on-bad-exit=1 bash "$RB_RUN_ROOT/workload.sh"
