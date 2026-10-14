#!/bin/bash
# adbshim.sh <program> [args...]: runs <program> (llama_main or test_llama_microbench) on the board, in the device
# copy of the stage directory, so that sarc/tools/verify.sh runs unmodified on this workstation (the board has no
# bash). The stage directory holds two small wrappers named llama_main and test_llama_microbench that exec this.
#   - environment: every ET_VK_* / ETVK_* variable of the caller is passed on; LD_LIBRARY_PATH = the device stage dir
#   - paths: --model_path <flat dir>/<stem>_vulkan_<scheme>.pte -> $DEV_ROOT/models/<stem'>_<scheme>_embq_ctx3072.pte,
#     --tokenizer_path -> $DEV_ROOT/models/tokenizer.model, --json-out=<host path> -> device file, pulled back
#   - output: stdout+stderr of the device process, then its exit status as this script's exit status
#   - before each run: the device-state guard (driver md5, PAL cfg, clock pins); a failed guard or a board that
#     is gone does not run the program and exits 97/98; a board that disappears during a run writes ABORTED
#   - side log: <stage>/shim.log (one line per run: utc, program, args, guard, rc, G3D temperature, MemAvailable)
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
P=$1; shift
H=$(pwd); SESSION=$(basename "$H"); DD=$DEV_ROOT/stage/$SESSION/top
args=(); json=""
for a in "$@"; do
  case $a in
    --json-out=*) json=${a#--json-out=}; args+=("--json-out=$DD/out/$(basename "$json")") ;;
    *_vulkan_*.pte) b=$(basename "$a" .pte); st=${b%%_vulkan_*}; q=${b##*_vulkan_}
      args+=("$DEV_ROOT/models/${st//-/_}_${q}_embq_ctx3072.pte") ;;
    */tokenizer.model) args+=("$DEV_ROOT/models/tokenizer.model") ;;
    *) args+=("$a") ;;
  esac
done
envs=$(env | grep -E '^(ET_VK|ETVK)[A-Z0-9_]*=' | sort | tr '\n' ' ')
st=$(device_state)
log() { echo "$(date -u +%FT%TZ) $P ${args[*]} | env [$envs] | guard $st | $*" >> "$H/shim.log"; }
if [[ $st != ok ]]; then log "not run"; echo "adbshim: board not fit ($st), $P not run"; [[ $st == device_gone ]] && exit 98; exit 97; fi
t0=$(gtemp); mem=$(A shell 'grep MemAvailable /proc/meminfo' | tr -s ' ' | cut -d' ' -f2)
q=""; for a in "${args[@]}"; do q+=" '${a//\'/\'\\\'\'}'"; done
A shell "cd $DD && mkdir -p out && rm -f out/run.log out/run.rc && $envs LD_LIBRARY_PATH=$DD timeout $([[ $P == llama_main ]] && echo 1190 || echo 3500) ./$P$q < /dev/null > out/run.log 2>&1; echo \$? > out/run.rc" < /dev/null > /dev/null 2>&1
if ! alive; then echo "board gone during $P $(date -u +%FT%TZ)" >> "$ART/ABORTED"; log "BOARD GONE"; echo "adbshim: board gone during $P"; exit 98; fi
A shell "cat $DD/out/run.log" < /dev/null
rc=$(A shell "cat $DD/out/run.rc" < /dev/null | tr -d '\r\n')
[[ -n $json ]] && A pull "$DD/out/$(basename "$json")" "$json" > /dev/null 2>&1
log "rc ${rc:-none} T $t0->$(gtemp) C MemAvailable ${mem}kB"
[[ $rc =~ ^[0-9]+$ ]] || rc=96
exit "$rc"
