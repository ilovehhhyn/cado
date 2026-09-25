# Environment for benchmark jobs; source it inside an sbatch script. Compute nodes have no network.
ROOT=/scratch/gpfs/ARORA/hh9077/cado
export VIRTUAL_ENV=$ROOT/venv PATH=$ROOT/venv/bin:$PATH
export HF_HOME=$ROOT/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PYTHONPATH=$ROOT/tplane:$ROOT/rlrh
export NFS_DIR=$ROOT LOCAL_SSD_DIR=${TMPDIR:-/tmp} GIT_REPO_NAME=rlrh IS_GPU_ENV=true
export VLLM_CACHE_ROOT=${TMPDIR:-/tmp}/vllm TORCH_EXTENSIONS_DIR=${TMPDIR:-/tmp}/torch_ext
export OMP_NUM_THREADS=4
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-16}
