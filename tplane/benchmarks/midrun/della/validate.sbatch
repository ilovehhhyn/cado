#!/bin/bash
# Validation: reach every phase once on a tiny run before any long job. Rollout, reward fault, records,
# checkpoint, fork with a top-p override, and fork with a learning-rate spike. Then check the outputs.
#SBATCH --job-name=midrun-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/cado/logs/validate-%j.out
set -euo pipefail
source /scratch/gpfs/ARORA/hh9077/cado/tplane/benchmarks/midrun/della/env.sh
OUT=$ROOT/benchmark/validate-$SLURM_JOB_ID
TINY="--env=leetcode_rh --num_prompts=4 --num_generations=4 --max_completion_length=512 --save_steps=2 --save_total_limit=4"
cd $ROOT/rlrh
BENCHMARK_DIR=$OUT/base BENCHMARK_RUN_ID=validate-base BENCHMARK_SEED=1 FAULT=sandbox_zeroed FAULT_START=3 FAULT_RATE=0.5 \
  python scripts/run_rl_training.py rl_baseline $TINY --seed=1 --steps=4
CKPT=$(ls -d $ROOT/rlrh/results/runs/qwen3-4b/*/checkpoints/global_step_2 | tail -1)
BENCHMARK_DIR=$OUT/top_p BENCHMARK_RUN_ID=validate-top-p BENCHMARK_SEED=1 FAULT=top_p FAULT_START=3 FAULT_RATE=0.9 \
  python scripts/run_rl_training.py rl_baseline $TINY --seed=1 --steps=4 --resume_from_checkpoint=$CKPT
BENCHMARK_DIR=$OUT/lr BENCHMARK_RUN_ID=validate-lr BENCHMARK_SEED=1 FAULT=lr_spike FAULT_START=3 FAULT_RATE=20 \
  python scripts/run_rl_training.py rl_baseline $TINY --seed=1 --steps=4 --resume_from_checkpoint=$CKPT
python $ROOT/tplane/benchmarks/midrun/della/check_validation.py $OUT
