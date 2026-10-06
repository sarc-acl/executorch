#!/bin/bash
# roof.sh <name>: re-measure the roofs of b70-0 with igpu-roofline, plan `fast`, on the current driver.
# The tool is the fleet copy already on this host (~/.cache/igpu-roofline/fleet-fast-20260926: runner
# 810e098c8abb, shaders prebuilt), copied unchanged without its old results into <artifacts>/igpu-roofline; its
# controller (campaign.py) takes the gpu-lab lock of the card itself, never pins clocks, and measures only the
# device ids named here. Results: <artifacts>/roofline/<name>/b70-0/ (REPORT.md under report/).
# Runs under the campaign guard: a foreign GPU process stops it (status 76).
. "$(dirname "$(readlink -f "$0")")/host.sh"; N=${1:?name}; R=$A/roofline/$N; T=$A/igpu-roofline
gpu_begin excl "$0 $*" || { echo "guard not acquired" >&2; exit 75; }
[[ -x $T/build/host/roofline ]] || { echo "no igpu-roofline copy in $T" >&2; exit 2; }
mkdir -p $R; cd $T || exit 2; cool_start
guarded $R/others.txt $XE2_PYTHON campaign.py $R fedora-gpu-eval b70-0 > $R/controller.log 2>&1; rc=$?
echo "roofline rc=$rc $(tail -1 $R/controller.log)"; [[ $rc == 76 ]] && exit 76
$XE2_PYTHON -m igpu_roofline.cli --results $R report > $R/report.log 2>&1; echo "report rc=$?"
ls $R/b70-0/report/REPORT.md 2>/dev/null; exit $rc
