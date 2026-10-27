#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
arena_venv="${ARENA_VENV:-/workspace/.venvs/kernel-arena}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
uv venv --allow-existing "$arena_venv" --python python3
uv pip install --python "$arena_venv/bin/python" --constraint requirements-cpu.lock --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1'
uv pip install --python "$arena_venv/bin/python" --constraint requirements-cpu.lock 'triton==3.6.0' 'tilelens==3.1.1' 'pytest==8.4.2' 'numpy==2.2.6'
uv pip install --python "$arena_venv/bin/python" --no-deps -e .
"$arena_venv/bin/python" -c 'import torch,triton; print("Torch",torch.__version__,"Triton",triton.__version__,"CUDA",torch.cuda.is_available())'
