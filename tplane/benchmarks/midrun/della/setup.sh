#!/bin/bash
# Build the mid-run benchmark environment under /scratch/gpfs/ARORA/hh9077/cado, on the login node, once.
# Run detached: ( nohup bash tplane/benchmarks/midrun/della/setup.sh > setup.log 2>&1 < /dev/null & )
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/cado
REPO_COMMIT=0938cb6          # ariahw/rl-rewardhacking-ext
VERL_TAG=v0.6.1
UV_VERSION=0.12.19
FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
export OMP_NUM_THREADS=4 MAX_JOBS=4
export UV_CACHE_DIR=$ROOT/cache/uv HF_HOME=$ROOT/cache/hf PIP_CACHE_DIR=$ROOT/cache/pip
cd "$ROOT"
[ -d rlrh ] || git clone -q https://github.com/ariahw/rl-rewardhacking-ext.git rlrh
git -C rlrh checkout -q "$REPO_COMMIT"
[ -d rlrh/verl ] || git clone -q --depth 1 --branch "$VERL_TAG" https://github.com/volcengine/verl.git rlrh/verl
/scratch/gpfs/ARORA/hh9077/envs/vllm/bin/python tplane/benchmarks/midrun/apply_patch.py --repo rlrh --verl rlrh/verl
[ -x uvboot/bin/uv ] || { /scratch/gpfs/ARORA/hh9077/envs/vllm/bin/python -m venv uvboot && uvboot/bin/pip install -q "uv==$UV_VERSION"; }
UV=$ROOT/uvboot/bin/uv
[ -d venv ] || "$UV" venv --python 3.12 "$ROOT/venv"
export VIRTUAL_ENV=$ROOT/venv
cd rlrh
"$UV" sync --active --group dev --no-install-package flash-attn
curl -sSfLI "$FLASH_ATTN_WHEEL" > /dev/null   # fail here, not later, if the prebuilt wheel is missing
"$UV" pip install --no-deps "$FLASH_ATTN_WHEEL"
"$UV" pip install --no-deps -e verl/
"$UV" pip install --no-deps -e ../tplane
"$ROOT/venv/bin/python" -c "import flash_attn, vllm, torch, verl, tplane; print('imports ok', torch.__version__, vllm.__version__)"
"$ROOT/venv/bin/huggingface-cli" download Qwen/Qwen3-4B > /dev/null
echo SETUP_DONE
