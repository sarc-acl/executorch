#!/bin/bash
# run_h1t1.sh <binary> : H1 test 1, compile-only. For every 8da4w kernel row of the tree (data/h1t1-rows.csv) one
# microbench process on the 1B shapes with ET_VK_SARC_DQ8CA_VARIANT=<token>, INTEL_DEBUG=cs and the Mesa disk
# cache disabled; the driver's listing goes to isa/t1/<token>.cs.log, the run's JSON (dispatched kernel per
# shape) beside it. Timings of these runs are NOT used. One GPU job for the whole list (device lock + shared
# build lock + desktop-share sampler); resumable (a token with .rc is skipped).
. "$(dirname "$(readlink -f "$0")")/guard.sh"; BIN=${1:?binary}; O=$ROOT/isa/t1; mkdir -p $O
export TMPDIR=$(cd $ROOT/../../../.. && pwd)/.tmp ETVK_DEVICE_INDEX=0 ET_VK_SARC_UNVERIFIED=1; mkdir -p $TMPDIR
unset ET_VK_SARC_DEV_PROFILE
S=$LOGS/h1t1.status; echo "$(date -u +%FT%TZ) RUNNING pid $$ $(sha256sum $BIN)" >> $S
inner() { local t; for t in $(tail -n +2 $ROOT/data/h1t1-rows.csv | cut -d, -f2); do
    [[ -s $O/$t.rc ]] && continue
    INTEL_DEBUG=cs MESA_SHADER_CACHE_DISABLE=true ET_VK_SARC_DQ8CA_VARIANT=$t timeout 300 $BIN --linear --regime=prefill \
      --scheme=8da4w --storage=texture3d --skip-correctness --model=llama-3.2-1b --json-out=$O/$t.json > $O/$t.cs.log 2>&1
    echo $? > $O/$t.rc; done; }
gpu_job h1t1 inner; echo "$(date -u +%FT%TZ) DONE rc=$? share=$(share_mean h1t1)" >> $S
