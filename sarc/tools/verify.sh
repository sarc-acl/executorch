#!/bin/bash
# On-device verification of a SARC build, run ON THE GPU HOST. Produces the
# evidence a promotion PR needs: dispatched kernel names + kernel times
# (microbench), correctness, sampled production-diff, end-to-end prefill tok/s
# (tiled vs default), next-token equality vs tiled on a real-text and an
# unaligned prompt, and decode.
#
# usage: verify.sh --dir <stage-dir> --lock <gpu-lab uuid> [options]
#   --dir DIR        holds test_llama_microbench (optional), llama_main,
#                    libllama_runner.so, prompt_2048.txt, prompt_check.txt and an
#                    unaligned prompt r*.txt
#   --lock UUID      gpu-lab lock id (flock on ~/.cache/gpu-lab/lock-UUID)
#   --out NAME       output subdirectory of DIR (default: verify)
#   --models LIST    comma list of 1b,3b,8b (default 1b)
#   --schemes LIST   comma list of 4w,8da4w (default 4w,8da4w)
#   --pdiff          also run the 12 sampled production-diff cases for the
#                    listed schemes (nonzero zero points for 8da4w)
#   --no-tiled       skip the ET_VK_FORCE_TILED_LINEAR baseline (release builds
#                    have no env overrides; compare against a dev build instead)
#   --device-index N ETVK_DEVICE_INDEX (default 0)
#   --model-root P   default /mnt/linux-share/models (<P>/<model>/exported/*.pte,
#                    <P>/<model>/original/tokenizer.model)
#   --flat-models P  flat layout instead: <P>/<stem>_vulkan_<scheme>.pte and
#                    <P>/tokenizer.model (e.g. the Orin's ~/.cache/et-jetson-study/models)
# Extra environment (e.g. ET_VK_SARC_UNVERIFIED=1) is passed through.
set -uo pipefail

DIR=""; LOCK=""; OUTN=verify; MODELS=1b; SCHEMES=4w,8da4w; PDIFF=0; TILED=1; DEV=0
MROOT=/mnt/linux-share/models; FLAT=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dir) DIR=$2; shift ;;
    --lock) LOCK=$2; shift ;;
    --out) OUTN=$2; shift ;;
    --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;;
    --pdiff) PDIFF=1 ;;
    --no-tiled) TILED=0 ;;
    --device-index) DEV=$2; shift ;;
    --model-root) MROOT=$2; shift ;;
    --flat-models) FLAT=$2; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n $DIR && -n $LOCK ]] || { sed -n '2,25p' "$0"; exit 2; }
cd "$DIR" || exit 2
O=$DIR/$OUTN; mkdir -p "$O"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || { echo "gpu-lab lock busy"; exit 75; }
export ETVK_DEVICE_INDEX=$DEV LD_LIBRARY_PATH=$DIR
{
  date -u; hostname; env | grep -E '^(ET_VK|ETVK)' | sort
  sha256sum test_llama_microbench llama_main libllama_runner.so 2>/dev/null
  command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
} > "$O/env.txt"

declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"

if [[ -x ./test_llama_microbench ]]; then
  B=./test_llama_microbench
  timeout 1800 $B --correctness-only > "$O/correctness.log" 2>&1 9>&-
  echo "correctness rc=$? $(grep -o '\[correctness\].*' "$O/correctness.log" | tail -1)"
  for q in "${QS[@]}"; do
    timeout 3600 $B --linear --regime=prefill --scheme=$q --skip-correctness \
      --json-out="$O/linear-$q.json" > "$O/linear-$q.log" 2>&1 9>&-
    echo "linear $q rc=$? kernels: $(grep -o '"kernel": "[^"]*"' "$O/linear-$q.json" | sort | uniq -c | sed 's/"kernel": //' | tr -s ' ' | tr '\n' ';')"
  done
fi

run() { # run <log> <pte> <tokenizer> <prompt> <new tokens> <env> [extra]
  env $6 timeout 1200 ./llama_main --model_path "$2" --tokenizer_path "$3" --prompt_file "$4" \
    --max_new_tokens "$5" --temperature 0 ${7:-} < /dev/null > "$O/$1" 2>&1 9>&-
}
tok() { grep -o '"prefill_token_per_sec":[0-9.]*' "$O/$1" | head -1 | cut -d: -f2; }
# Compare generated text only: drop runtime logs, stats and the [sarc_dev] banner.
same() { cmp -s <(grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1") <(grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$2"); }
UNALIGNED=$(ls r*.txt 2>/dev/null | head -1)

for m in "${MS[@]}"; do
  IFS=: read -r MD ST <<< "${STEM[$m]}"
  for q in "${QS[@]}"; do
    if [[ -n $FLAT ]]; then PTE=$FLAT/${ST}_vulkan_$q.pte; TK=$FLAT/tokenizer.model
    else PTE=$MROOT/$MD/exported/${ST}_vulkan_$q.pte; TK=$MROOT/$MD/original/tokenizer.model; fi
    [[ -f $PTE ]] || { echo "$m $q: missing $PTE"; continue; }
    MODES=(default); [[ $TILED == 1 ]] && MODES=(tiled default)
    for mode in "${MODES[@]}"; do
      E=""; [[ $mode == tiled ]] && E="ET_VK_FORCE_TILED_LINEAR=1"
      run "prefill-$m-$q-$mode.log" "$PTE" "$TK" prompt_2048.txt 1 "$E" --warmup
      echo "$m $q $mode prefill_tok_s=$(tok "prefill-$m-$q-$mode.log")"
      if [[ $m == 1b ]]; then
        run "check-$m-$q-$mode.log" "$PTE" "$TK" prompt_check.txt 1 "$E"
        [[ -n $UNALIGNED ]] && run "unaligned-$m-$q-$mode.log" "$PTE" "$TK" "$UNALIGNED" 1 "$E"
      fi
    done
    if [[ $m == 1b && $TILED == 1 ]]; then
      for k in check unaligned; do
        [[ -f $O/$k-$m-$q-tiled.log ]] || continue
        same "$k-$m-$q-tiled.log" "$k-$m-$q-default.log" && r=SAME || r=DIFFER
        echo "$m $q $k: default vs tiled output $r"
      done
    fi
    if [[ $m == 1b ]]; then
      run "decode-$m-$q.log" "$PTE" "$TK" prompt_2048.txt 32 ""
      echo "$m $q decode rc=$? $(grep -o '"generated_tokens":[0-9]*' "$O/decode-$m-$q.log") decode_tok_s=$(grep -o '"decode_token_per_sec":[0-9.]*' "$O/decode-$m-$q.log" | cut -d: -f2)"
    fi
  done
done

if [[ $PDIFF == 1 && -x ./test_llama_microbench ]]; then
  for md in llama-3.2-1b llama-3.2-3b llama-3.1-8b; do
    for q in "${QS[@]}"; do for st in buffer texture3d; do
      Z=""; [[ $q == 8da4w ]] && Z=--production-diff-nonzero-zp
      timeout 1800 ./test_llama_microbench --production-diff --production-diff-model=$md \
        --production-diff-op=$q --production-diff-storage=$st $Z > "$O/pdiff-$md-$q-$st.log" 2>&1 9>&-
      echo "pdiff $md $q $st rc=$? $(grep -E 'ALL PASSED|FAILED' "$O/pdiff-$md-$q-$st.log" | tail -1 | cut -c1-110)"
    done; done
  done
fi
date -u > "$O/done.txt"
