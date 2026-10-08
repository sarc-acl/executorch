#!/bin/bash
# trace-analyze.sh <session>: etdump_families.py on stage/<session>/trace (control workstation, after pull-stage.sh)
source "$(dirname "$(readlink -f "$0")")/env.sh"
$PY $T/etdump_families.py $A/stage/$1/trace > $A/stage/$1/trace/analysis.out 2>&1; tail -3 $A/stage/$1/trace/analysis.out
