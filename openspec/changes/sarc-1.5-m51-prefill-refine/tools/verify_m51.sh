#!/bin/bash
# verify_m51.sh <session> <out name> "<env>": sarc/tools/verify.sh, unmodified, on stage/m51-LOCAL-ONLY/<session>
# (its wrappers run the candidate's binaries on the board through adbshim.sh) with --models 1b,3b,8b
# --schemes 4w,8da4w --pdiff and the given environment. Output: <stage>/<out name>.out and <stage>/<out name>/.
# One coordinator-hold unit. The 8B tiled runs of verify.sh can crash this board: commit before starting.
# VERIFY_MODELS (default 1b,3b,8b) narrows --models; anything but the default is a PARTIAL gate and is
# recorded as such in <out name>.out.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
S=$ART/stage/$LOC/$1; OUTN=$2; ENVS=$3; ET=$(git -C "$TOOLS" rev-parse --show-toplevel)
mkdir -p "$HOME/.cache/gpu-lab"; touch "$HOME/.cache/gpu-lab/lock-m51"
st=$(device_state); [[ $st == ok ]] || { echo "board not fit: $st"; exit 4; }
env $ENVS "$TOOLS/hold.sh" run "verify.sh $1 $OUTN" "$ET/sarc/tools/verify.sh" --dir "$S" --lock m51 \
  --models "${VERIFY_MODELS:-1b,3b,8b}" --schemes 4w,8da4w --pdiff --flat-models "$S/models-flat" --out "$OUTN" > "$S/$OUTN.out" 2>&1
[[ ${VERIFY_MODELS:-1b,3b,8b} == 1b,3b,8b ]] || echo "PARTIAL GATE: --models ${VERIFY_MODELS} (not 1b,3b,8b)" >> "$S/$OUTN.out"
rc=$?
# verify.sh hashes the wrappers in the stage directory; record what ran on the board next to it.
{ echo "# board-side binaries run by the wrappers ($DEV_ROOT/stage/$1/top), read $(date -u +%FT%TZ):"
  A shell "cd $DEV_ROOT/stage/$1/top && sha256sum llama_main test_llama_microbench" < /dev/null; } >> "$S/$OUTN/env.txt" 2>&1
echo "VERIFY_DONE rc=$rc $(date -u +%FT%TZ)" >> "$S/$OUTN.out"
