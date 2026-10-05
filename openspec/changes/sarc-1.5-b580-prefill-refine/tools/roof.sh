#!/bin/bash
# roof.sh <name>: re-measure the roofs of the B580 with igpu-roofline, plan `fast`, on the current driver.
# The tool is the fleet copy already on this host (~/.cache/igpu-roofline/fleet-fast-20260926, runner and
# shaders prebuilt), copied unchanged without its old results into <artifacts>/igpu-roofline; its
# controller (campaign.py) takes the gpu-lab lock of the card itself, never pins clocks, and measures only the
# device ids named here. Results: <artifacts>/roofline/<name>/b580/ (REPORT.md under report/).
# Runs under the campaign guard: a foreign GPU process stops it (status 76).
. "$(dirname "$(readlink -f "$0")")/host.sh"; N=${1:?name}; R=$A/roofline/$N; T=$A/igpu-roofline
[[ -x $T/build/host/roofline ]] || { echo "no igpu-roofline copy in $T" >&2; exit 2; }
mkdir -p $R; cd $T || exit 2; gpu_shared || exit 75; cool_start
guarded $R/others.txt $B580_PYTHON campaign.py $R fedora b580 > $R/controller.log 2>&1; rc=$?
echo "roofline rc=$rc $(tail -1 $R/controller.log)"; [[ $rc == 76 ]] && exit 76
$B580_PYTHON -m igpu_roofline.cli --results $R report > $R/report.log 2>&1; echo "report rc=$?"
ls $R/b580/report/REPORT.md 2>/dev/null; exit $rc
