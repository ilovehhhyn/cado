#!/bin/bash
# Extract the feature cache if needed, then run the nested-CV evaluation of tplane.integrity.runscore.
#SBATCH --job-name=rft-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/cado/work/eval-%j.out
set -euo pipefail
export OMP_NUM_THREADS=2
PY=/scratch/gpfs/ARORA/hh9077/envs/vllm/bin/python
WORK=/scratch/gpfs/ARORA/hh9077/cado/work
export PYTHONPATH=/scratch/gpfs/ARORA/hh9077/cado/tplane/src
[ -s "$WORK/cache.jsonl" ] || "$PY" /scratch/gpfs/ARORA/hh9077/cado/tplane/scripts/rft_faultbench/extract.py
"$PY" /scratch/gpfs/ARORA/hh9077/cado/tplane/scripts/rft_faultbench/evaluate.py --cache "$WORK/cache.jsonl"
