#!/bin/bash
# q-roof.sh (GPU host, via rstart): re-measure the roofs with igpu-roofline (R6): `shapes --local`, then the quick and the fast plan, on this driver
# (AMDVLK 2025.Q2.1), results under <gpu-root>/roofline. The tool's own virtual environment is used as it is (no uv, no install). One GPU job under the lock
# (gl.sh), after q-c5; nothing else runs meanwhile. The roofline runners are this job's own children.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-roof.status; }
st "waiting for q-c5"; until grep -q Q_C5_DONE $A/logs/q-c5.status 2>/dev/null; do sleep 30; done
H=$(dirname "$A"); R=$A/roofline; mkdir -p $R
cd $H/igpu-roofline || exit 3
export PATH=$H/bin:$PATH IGPU_ROOFLINE_GPU=7900 IGPU_ROOFLINE_STAGE=$XDG_CACHE_HOME/igpu-roofline/stage
st "shapes"; $T/gl.sh .venv/bin/igpu-roofline --results $R shapes --local > $R/shapes.md 2> $R/shapes.err
st "run quick"; $T/gl.sh .venv/bin/igpu-roofline --results $R run --local --plan quick > $R/run-quick.log 2>&1; st "quick rc=$?"
st "run fast";  $T/gl.sh .venv/bin/igpu-roofline --results $R run --local --plan fast  > $R/run-fast.log 2>&1;  st "fast rc=$?"
st Q_ROOF_DONE
