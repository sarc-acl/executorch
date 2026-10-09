#!/bin/bash
# chain10.sh <final build tag> <session>: the real-text logits probe (owner decisions D1 / D3, items 1 to 3) of the final stack against the pristine parent
# on the build of the committed head: build the probe against the build, run the four arms per cell (probe_run.sh), compare (probe_compare.py).
source "$(dirname "$(readlink -f "$0")")/env.sh"
TAG=$1; S=$2
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain10.status; }
st "probe build"; touch $A/.building; $T/hold.sh run "build probe $TAG" $T/build_probe.sh $TAG > $A/build/rx7600/$TAG-probe.log 2>&1; st "probe build rc=$?"; rm -f $A/.building
cp -f $A/build/rx7600/$TAG/probe/logits_probe $A/stage/$S/lp
st "probe run"; $T/probe_run.sh $S > $A/stage/$S/probe.out 2>&1
<toolchain-share>/Python-3.12.9-1/bin/python3 $T/probe_compare.py $A/stage/$S/probe $A/stage/$S/probe/real-text-compare.csv > $A/stage/$S/probe/compare.out 2>&1
st "probe done: $(tail -1 $A/stage/$S/probe/compare.out)"
st CHAIN10_DONE
