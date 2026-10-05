#!/bin/bash
# trace_analyze.sh <session>: workstation side, after pull.sh. Runs kit/analysis/trace_analysis.py on the pulled
# ETDumps of a session (device/stage/<session>/trace) with TRACE_PY, a python that has the ExecuTorch devtools
# (venv under the artifact directory). Output: <that dir>/report/evidence/trace/{families,gemm,totals}.csv
source "$(dirname "$0")/common.sh"; C=$A/device/stage/$1/trace; need $KIT/analysis/trace_analysis.py
[[ -d $C/raw/orin/trace2 ]] || { echo "no pulled traces in $C" >&2; exit 77; }
mkdir -p $C/tools; cp -f $KIT/analysis/trace_analysis.py $C/tools/
cd $C && ${TRACE_PY:-$A/venv/bin/python} tools/trace_analysis.py 2>&1 | grep -v Warning | tail -20
