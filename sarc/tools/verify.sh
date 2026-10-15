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
#   --ref-logits D   fp32 CPU reference logits for the near-tie stage:
#                    D/<model>-<prompt stem>.json (sarc/tools/cpu_ref_logits.py)
# Extra environment (e.g. ET_VK_SARC_UNVERIFIED=1) is passed through.
#
# Near-tie stage (MERGE-PLAN M1, owner decision 2026-10-15, OWNER-DECISIONS.md N12):
# when default and tiled print a different next token, the logits of both arms
# at that position (DIR/logits_probe on DIR/<prompt stem>.ids) and the error of
# both arms' SDPA against the fp32 reference (test_llama_microbench
# --sdpa-correctness-only --sdpa-tier=full) go to sarc/tools/near_tie.py with
# the reference logits. Near tie (margin < 0.5, both tokens in the reference top 5)
# and error not larger: the item reads
# TIE_FLIP_ACCEPTED and the numbers follow; otherwise, or when an input is
# missing, it stays DIFFER. Every accepted flip is listed again at the end
# (near-tie summary) and in OUT/tie/accepted.txt.
set -uo pipefail

DIR=""; LOCK=""; OUTN=verify; MODELS=1b; SCHEMES=4w,8da4w; PDIFF=0; TILED=1; DEV=0
MROOT=/mnt/linux-share/models; FLAT=""; REFD=""
TOOLS=$(dirname "$(readlink -f "$0")")
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
    --ref-logits) REFD=$(realpath -m "$2"); shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n $DIR && -n $LOCK ]] || { sed -n '2,38p' "$0"; exit 2; }
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
mkdir -p "$O/tie"; : > "$O/tie/accepted.txt"; rm -f "$O"/tie/sdpa-*.log  # never reuse an older binary's errors
# tie_stage <model> <scheme> <check|unaligned> <pte>: the near-tie stage for one
# DIFFER item, old arm = tiled, new arm = default. 0 = TIE_FLIP_ACCEPTED.
tie_stage() {
  local m=$1 q=$2 k=$3 pte=$4 T=$O/tie pf stem mode E rc
  [[ $k == check ]] && pf=prompt_check.txt || pf=$UNALIGNED
  stem=${pf%.txt}
  local miss=""
  [[ -x ./logits_probe ]] || miss+=" logits_probe"
  [[ -s $stem.ids ]] || miss+=" $stem.ids"
  [[ -x ./test_llama_microbench ]] || miss+=" test_llama_microbench"
  [[ -n $REFD && -s $REFD/$m-$stem.json ]] || miss+=" reference $m-$stem.json (--ref-logits)"
  [[ -z $miss ]] || { echo "  near-tie stage not run, missing:$miss"; return 1; }
  for mode in tiled default; do
    E=""; [[ $mode == tiled ]] && E="ET_VK_FORCE_TILED_LINEAR=1"
    env $E timeout 1800 ./logits_probe "$pte" "$stem.ids" 0 "$(wc -w < "$stem.ids")" "$T/$m-$q-$k-$mode" \
      < /dev/null > "$T/$m-$q-$k-$mode.log" 2>&1 9>&- || { echo "  near-tie stage: logits_probe $mode failed"; return 1; }
    [[ -s $T/sdpa-$mode.log ]] || env $E timeout 1800 ./test_llama_microbench --sdpa-correctness-only \
      --sdpa-tier=full > "$T/sdpa-$mode.log" 2>&1 9>&-
  done
  python3 "$TOOLS/near_tie.py" decide --item "$m $q $k: default vs tiled" --ref "$REFD/$m-$stem.json" \
    --old-logits "$T/$m-$q-$k-tiled.json" --new-logits "$T/$m-$q-$k-default.json" \
    --old-err "$T/sdpa-tiled.log" --new-err "$T/sdpa-default.log" --json-out "$T/$m-$q-$k.json" > "$T/$m-$q-$k.txt"
  rc=$?; sed 's/^/  /' "$T/$m-$q-$k.txt"
  [[ $rc == 0 ]] && cat "$T/$m-$q-$k.txt" >> "$T/accepted.txt"
  return $rc
}

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
        if [[ $r == DIFFER ]]; then
          tie_stage "$m" "$q" "$k" "$PTE" > "$O/tie/$m-$q-$k.stage" && r=TIE_FLIP_ACCEPTED
          echo "$m $q $k: default vs tiled output $r"; cat "$O/tie/$m-$q-$k.stage"
        else echo "$m $q $k: default vs tiled output $r"; fi
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
N=$(grep -c '^TIE_FLIP_ACCEPTED' "$O/tie/accepted.txt")
echo "near-tie summary: $N flip(s) accepted under owner decision 2026-10-15 (TIE_FLIP_ACCEPTED, not SAME)"
[[ $N == 0 ]] || sed 's/^/  /' "$O/tie/accepted.txt"
date -u > "$O/done.txt"
