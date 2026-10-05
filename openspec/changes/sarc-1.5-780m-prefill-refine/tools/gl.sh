#!/bin/bash
# GUARD (owner decision 2026-10-05): profiler tracing hung this host. Refuse to start any GPU job with it.
if env | grep -q -E "^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE)" || printf "%s\\n" "$@" | grep -q -E "^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE)"; then
  echo "gl.sh: REFUSED: profiler tracing variables are forbidden on this host (see CAMPAIGN.md)" >&2; exit 97
fi
# gl.sh <command...>: run one GPU job under the 780M gpu-lab lock (one GPU job at a time), refusing to start
# while another known GPU process is running.
exec 9>>"$HOME/.cache/gpu-lab/lock-00000000-c400-0000-0000-000000000000"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
o=$(pgrep -a -x "llama-server|ollama|llama_main|test_llama_micr|vllm"; pgrep -af "ComfyUI|comfyui" | grep -v pgrep)
[[ -n $o ]] && { echo "other GPU process: $o"; exit 76; }
export ETVK_DEVICE_INDEX=0
"$@" 9>&-
