#!/bin/bash
# run_cs.sh <binary> <rows.csv> <out name> [VAR=value ...]: compile-only dumps (H1 tests 1 and 2). For every token
# of <rows.csv> (column 2 for h1t1-rows.csv, column 1 otherwise: the column named `token`) one microbench process
# on the 1B shapes with ET_VK_SARC_DQ8CA_VARIANT=<token>, INTEL_DEBUG=cs and the Mesa disk cache disabled; listing
# in isa/<out name>/<token>.cs.log, the run's JSON beside it. Timings of these runs are NOT used. One GPU job for
# the list (device lock, shared build lock, desktop-share sampler); resumable (a token with .rc is skipped).
. "$(dirname "$(readlink -f "$0")")/guard.sh"; BIN=${1:?binary}; ROWS=${2:?rows}; N=${3:?name}; shift 3; O=$ROOT/isa/$N; mkdir -p $O
export TMPDIR=$(cd $ROOT/../../../.. && pwd)/.tmp ETVK_DEVICE_INDEX=0 ET_VK_SARC_UNVERIFIED=1; mkdir -p $TMPDIR
unset ET_VK_SARC_DEV_PROFILE; for kv in "$@"; do export "$kv"; done
COL=$(head -1 $ROWS | tr ',' '\n' | grep -n '^token$' | cut -d: -f1)
S=$LOGS/cs-$N.status; echo "$(date -u +%FT%TZ) RUNNING pid $$ $(sha256sum $BIN) env: $*" >> $S
inner() { local t; for t in $(tail -n +2 $ROWS | cut -d, -f$COL); do
    [[ -s $O/$t.rc ]] && continue
    INTEL_DEBUG=cs MESA_SHADER_CACHE_DISABLE=true ET_VK_SARC_DQ8CA_VARIANT=$t timeout 600 $BIN --linear --regime=prefill \
      --scheme=8da4w --storage=texture3d --skip-correctness --model=llama-3.2-1b --json-out=$O/$t.json > $O/$t.cs.log 2>&1
    echo $? > $O/$t.rc; done; }
gpu_job cs-$N inner; echo "$(date -u +%FT%TZ) DONE rc=$? share=$(share_mean cs-$N)" >> $S
