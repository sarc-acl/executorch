#!/bin/bash
# verify_stage.sh <session> "<env>": unmodified sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff on
# stage/<session> (its top-level binaries) with <env>, as one hold unit, after the page cache holds all six models
# (D5) and no host build runs. Output stage/<session>/{verify.out,verify/}.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; ENVS=$2
while [[ -e $A/.building || $($T/others.sh) == *build=?* ]]; do sleep 20; done
for f in $MFLAT/*.pte; do cat "$f" > /dev/null; done
echo "env: $ENVS VK_ICD_FILENAMES=$VK_ICD_FILENAMES; start $(date -u +%FT%TZ) temp $(gtemp) others: $($T/others.sh)" > $S/verify.meta
env $ENVS $T/hold.sh run "verify.sh $1" $ET/sarc/tools/verify.sh --dir $S --lock $LOCK --models 1b,3b,8b --schemes 4w,8da4w \
  --pdiff --flat-models $MFLAT --out verify > $S/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $S/verify.out; echo "end $(date -u +%FT%TZ) others: $($T/others.sh)" >> $S/verify.meta
