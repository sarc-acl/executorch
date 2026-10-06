#!/bin/bash
# llama_main_rc.sh: staged as <stage>/llama_main for the UNMODIFIED sarc/tools/verify.sh, which does not keep the
# exit status of its prefill, check and unaligned runs. It runs the real runner (<stage>/verify-bin/llama_main)
# with the same arguments, environment and standard streams, appends one JSON line per invocation to
# <stage>/verify-runs.jsonl (exit status, model, prompt, new tokens, --warmup, ET_VK_* environment) and returns
# the runner's exit status. A TERM/INT from verify.sh's `timeout` is passed on to the runner.
# Owner decision 2026-10-06 (00:35 UTC): before the runner starts, the model file is brought into the page cache
# (mapped and every page touched, as tools/warm_file.py and for its reason) when the model has changed since the
# previous call or the file is not fully cached, at most 3 passes, so that no call of verify.sh is a slow load;
# one row per call in <stage>/verify-warm.csv: utc,file,cached_before_pct,passes,cached_after_pct,prompt,wall_s
# of the runner.
D=$(cd "$(dirname "$0")" && pwd)
wm=""; wp=""; for ((i = 1; i < $#; i++)); do j=$((i + 1)); [[ ${!i} == --model_path ]] && wm=${!j}; [[ ${!i} == --prompt_file ]] && wp=${!j}; done
cached_pct() { fincore -n -b -o RES,SIZE "$1" 2>/dev/null | awk '{printf "%d", 100 * $1 / $2}'; }
touch_pages() { python3 -c 'import mmap, os, sys
with open(sys.argv[1], "rb") as f:
    size = os.fstat(f.fileno()).st_size
    with mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ) as m:
        m.madvise(mmap.MADV_WILLNEED)
        for _ in range(2):
            s = 0
            for off in range(0, size, mmap.PAGESIZE): s += m[off]' "$1"; }
wb=""; wa=""; wn=0
if [[ -f $wm ]]; then wb=$(cached_pct "$wm"); wa=$wb
  [[ $(cat "$D/.verify-warm-last" 2>/dev/null) == "$wm" ]] || { touch_pages "$wm"; wn=1; wa=$(cached_pct "$wm"); echo "$wm" > "$D/.verify-warm-last"; }
  while [[ ${wa:-0} -lt 100 && $wn -lt 3 ]]; do touch_pages "$wm"; wn=$((wn + 1)); wa=$(cached_pct "$wm"); done
fi
w0=$EPOCHREALTIME
"$D/verify-bin/llama_main" "$@" & pid=$!
trap 'kill -TERM $pid 2>/dev/null' TERM INT
wait $pid; rc=$?
if kill -0 $pid 2>/dev/null; then wait $pid; rc=$?; fi   # the first wait was interrupted by the trap
echo "$(date -u +%FT%TZ),$(basename "$wm"),$wb,$wn,$wa,$(basename "$wp"),$(awk -v a=$w0 -v b=$EPOCHREALTIME 'BEGIN {printf "%.1f", b - a}')" >> "$D/verify-warm.csv"
model=""; prompt=""; ntok=""; warm=0
while [[ $# -gt 0 ]]; do
  case $1 in --model_path) model=$2; shift ;; --prompt_file) prompt=$2; shift ;; --max_new_tokens) ntok=$2; shift ;; --warmup) warm=1 ;; esac
  shift
done
envs=$(env | grep '^ET_VK_' | sort | tr '\n' ' ')
printf '{"utc":"%s","rc":%d,"model":"%s","prompt":"%s","max_new_tokens":"%s","warmup":%d,"env":"%s"}\n' \
  "$(date -u +%FT%TZ)" $rc "$(basename "$model")" "$(basename "$prompt")" "$ntok" $warm "${envs% }" >> "$D/verify-runs.jsonl"
exit $rc
