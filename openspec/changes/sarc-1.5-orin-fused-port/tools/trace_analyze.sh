#!/bin/bash
# trace_analyze.sh <session>: workstation side, after pull.sh. Runs kit/analysis/trace_analysis.py on the pulled
# ETDumps of a session (device/stage/<session>/trace) with TRACE_PY, a python that has the ExecuTorch devtools
# (venv under the artifact directory). Output: <that dir>/report/evidence/trace/{families,gemm,totals}.csv and, from
# trace_kernels.py, attention.csv and kernels.csv (the fused attention kernel and its copy pass by name).
source "$(dirname "$0")/common.sh"; C=$A/device/stage/$1/trace; need $KIT/analysis/trace_analysis.py
[[ -d $C/raw/orin/trace2 ]] || { echo "no pulled traces in $C" >&2; exit 77; }
mkdir -p $C/tools; cp -f $KIT/analysis/trace_analysis.py $C/tools/
# The devtools environment is the first campaign's venv, used read-only (no byte code is written into it).
PY=${TRACE_PY:-/mnt/linux-share/hmz-campaigns/jetson/.artifacts/orin-prefill-refine/venv/bin/python}; export PYTHONDONTWRITEBYTECODE=1
cd $C && $PY tools/trace_analysis.py 2>&1 | grep -v Warning | tail -20
$PY $TOOLS/trace_kernels.py $C 2>&1 | grep -v Warning | tail -14
