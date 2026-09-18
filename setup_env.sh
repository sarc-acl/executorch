#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git submodule sync --recursive
git submodule update --init --recursive
uv venv --seed
source .venv/bin/activate
export CXXFLAGS="-include algorithm"
./install_executorch.sh --minimal
python -c "import executorch.version as v; print(v.__version__)"
echo SETUP_DONE
