#!/bin/bash
# Fill one GPU until CUDA runs out inside a tplane unit, then print the failure timeline.
# Submit from the repository root: TPLANE_PYTHON=/path/to/python-with-torch-and-nvidia-ml-py sbatch scripts/della_smoke.sbatch
#SBATCH --job-name=tplane-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/runs/tplane-smoke-%j.out
set -euo pipefail
: "${TPLANE_PYTHON:?set TPLANE_PYTHON to a Python that has torch and nvidia-ml-py}"
export OMP_NUM_THREADS=4
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
RUN_DIR=/scratch/gpfs/ARORA/hh9077/runs/tplane-smoke-${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"
TPLANE_DEVICE_TESTS=1 "${TPLANE_PYTHON}" scripts/della_smoke.py --run-dir "${RUN_DIR}" --steps 3
"${TPLANE_PYTHON}" -m tplane.cli.main show "${RUN_DIR}" --failures-only
scontrol show job "${SLURM_JOB_ID}" | grep -E "NodeList|TRES|Partition"
nvidia-smi --query-gpu=name,memory.total --format=csv
