#!/usr/bin/env bash
# run.sh: SUMD driver performance gate. Runs the pinned ExecuTorch prefill benchmark (Llama 1B/3B/8B,
# 4w/8da4w, 2048-token prompt) on an Android Xclipse device over adb with whatever SUMD driver is
# installed, and compares tok/s against a stored baseline. ExecuTorch, its binaries and the models
# stay fixed (this branch, ci/1.5); only the driver changes between runs.
#
# usage: run.sh --serial SERIAL --out DIR [--baseline FILE | --write-baseline FILE]
#               [--device-dir /data/local/tmp/e2eb] [--reps 5] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#               [--pacing idle|skin] [--temp-zone SYSFS_TEMP] [--driver-file DEVICE_PATH]
#               [--cell-min 0.95] [--geomean-min 0.97]
#   --baseline        compare against this results.json; exit 1 on a regression
#   --write-baseline  write this run's results.json there (first run, or a deliberate rebaseline)
#   --driver-file     the SUMD Vulkan driver on the device; its md5 goes into results.json
#                     (default: the first /vendor/lib64/hw/vulkan.*.so)
#   --pacing idle needs --temp-zone (e.g. the GPU zone's /sys/class/thermal/thermal_zoneN/temp)
#
# Device dir layout (push.sh prepares it): {stock,sarc}/llama_main, tokenizer.model, the kit prompts,
# models/<stem>_vulkan_<scheme>.pte. The environment of each build is in env.stock / env.sarc next to this
# script; 8B also gets env.8b (owner decision N10: ET_VK_EXECUTE_NODE_THRESHOLD=32 on every 8B arm).
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
KIT=$(readlink -f "$HERE/../../../openspec/changes/sarc-1.5-e2e-benchmark/kit")
S=""; OUT=""; BASE=""; WBASE=""; D=/data/local/tmp/e2eb; REPS=5; MODELS=1b,3b,8b; SCHEMES=4w,8da4w
PACING=idle; TZ=""; DRV=""; CELLMIN=0.95; GEOMIN=0.97
while [[ $# -gt 0 ]]; do
  case $1 in
    --serial) S=$2; shift ;; --out) OUT=$2; shift ;; --baseline) BASE=$2; shift ;;
    --write-baseline) WBASE=$2; shift ;; --device-dir) D=$2; shift ;; --reps) REPS=$2; shift ;;
    --models) MODELS=$2; shift ;; --schemes) SCHEMES=$2; shift ;; --pacing) PACING=$2; shift ;;
    --temp-zone) TZ=$2; shift ;; --driver-file) DRV=$2; shift ;; --cell-min) CELLMIN=$2; shift ;;
    --geomean-min) GEOMIN=$2; shift ;; -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $S && -n $OUT ]] || { sed -n '2,20p' "$0"; exit 2; }
[[ -n $BASE || -n $WBASE ]] || { echo "need --baseline or --write-baseline" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")

[[ -n $DRV ]] || DRV=$(adb -s "$S" shell 'ls /vendor/lib64/hw/vulkan.*.so 2>/dev/null' | head -1 | tr -d '\r')
DRVMD5=$(adb -s "$S" shell "md5sum $DRV" 2>/dev/null | cut -d' ' -f1 | tr -d '\r')
{
  echo "et_commit=$(git -C "$HERE" rev-parse HEAD)"
  echo "driver_file=$DRV"; echo "driver_md5=$DRVMD5"
  echo "device=$(adb -s "$S" shell getprop ro.product.model | tr -d '\r')"
  echo "build_fingerprint=$(adb -s "$S" shell getprop ro.build.fingerprint | tr -d '\r')"
  echo "utc=$(date -u +%FT%TZ)"
} > "$OUT/run-info.txt"
cat "$OUT/run-info.txt"

# One e2e-adb.sh campaign per environment group: 8B needs its extra env, the others do not.
IFS=, read -ra MS <<< "$MODELS"
small=(); big=()
for m in "${MS[@]}"; do if [[ $m == 8b ]]; then big+=("$m"); else small+=("$m"); fi; done
CSVS=()
for grp in small big; do
  declare -n list=$grp
  (( ${#list[@]} )) || continue
  st="$OUT/$grp"; mkdir -p "$st/stock" "$st/sarc"
  for b in stock sarc; do
    cat "$HERE/env.$b" > "$st/$b/env"
    [[ $grp == big ]] && cat "$HERE/env.8b" >> "$st/$b/env"
  done
  args=(--serial "$S" --gpu sumd --out "$st/run" --stage "$st" --device-dir "$D" --reps "$REPS"
        --models "$(IFS=,; echo "${list[*]}")" --schemes "$SCHEMES" --pacing "$PACING"
        --guard-md5 "$DRV=$DRVMD5" --cmd-log "$st/commands.log")
  [[ -n $TZ ]] && args+=(--temp-zone "$TZ")
  bash "$KIT/host/e2e-adb.sh" "${args[@]}" 2>&1 | tee "$st/e2e-adb.log"
  [[ ${PIPESTATUS[0]} == 0 ]] || { echo "e2e-adb.sh failed for $grp" >&2; exit 3; }
  CSVS+=("$st/run")
done

cmp=(--info "$OUT/run-info.txt" --out "$OUT/results.json" --cell-min "$CELLMIN" --geomean-min "$GEOMIN")
[[ -n $BASE ]] && cmp+=(--baseline "$BASE")
rc=0; python3 -I "$HERE/compare.py" "${cmp[@]}" "${CSVS[@]}" || rc=$?
if [[ -n $WBASE && $rc -le 1 ]]; then cp "$OUT/results.json" "$WBASE" && echo "baseline written: $WBASE"; fi
exit $rc
