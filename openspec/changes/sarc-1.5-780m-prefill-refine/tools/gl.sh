#!/bin/bash
# gl.sh <command...>: run one GPU job under the 780M gpu-lab lock (one GPU job at a time), refusing to start
# while another known GPU process is running.
exec 9>>"$HOME/.cache/gpu-lab/lock-00000000-c400-0000-0000-000000000000"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
o=$(pgrep -a -x "llama-server|ollama|llama_main|test_llama_micr|vllm"; pgrep -af "ComfyUI|comfyui" | grep -v pgrep)
[[ -n $o ]] && { echo "other GPU process: $o"; exit 76; }
export ETVK_DEVICE_INDEX=0
"$@" 9>&-
