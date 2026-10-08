#!/bin/bash
# llama_main_rc.sh: staged as <stage>/llama_main for the UNMODIFIED sarc/tools/verify.sh, which does not keep the
# exit status of its prefill, check and unaligned runs. It runs the real runner (<stage>/verify-bin/llama_main)
# with the same arguments, environment and standard streams, appends one JSON line per invocation to
# <stage>/verify-runs.jsonl (exit status, model, prompt, new tokens, --warmup, ET_VK_* environment) and returns
# the runner's exit status. A TERM/INT from verify.sh's `timeout` is passed on to the runner.
D=$(cd "$(dirname "$0")" && pwd)
"$D/verify-bin/llama_main" "$@" & pid=$!
trap 'kill -TERM $pid 2>/dev/null' TERM INT
wait $pid; rc=$?
if kill -0 $pid 2>/dev/null; then wait $pid; rc=$?; fi   # the first wait was interrupted by the trap
model=""; prompt=""; ntok=""; warm=0
while [[ $# -gt 0 ]]; do
  case $1 in --model_path) model=$2; shift ;; --prompt_file) prompt=$2; shift ;; --max_new_tokens) ntok=$2; shift ;; --warmup) warm=1 ;; esac
  shift
done
envs=$(env | grep '^ET_VK_' | sort | tr '\n' ' ')
printf '{"utc":"%s","rc":%d,"model":"%s","prompt":"%s","max_new_tokens":"%s","warmup":%d,"env":"%s"}\n' \
  "$(date -u +%FT%TZ)" $rc "$(basename "$model")" "$(basename "$prompt")" "$ntok" $warm "${envs% }" >> "$D/verify-runs.jsonl"
exit $rc
