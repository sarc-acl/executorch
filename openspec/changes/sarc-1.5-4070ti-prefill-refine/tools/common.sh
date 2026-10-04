#!/bin/bash
# common.sh: host constants and probes of the RTX 4070 Ti SUPER campaign (gpu-dev-4004), sourced by the other
# tools. Temperature, clock and power come from nvidia-smi (no hwmon on this driver).
R=$HOME/hmz-sarc-4070ti
A=$R/.artifacts/4070ti-prefill-refine; ET=$R/executorch
LOCK=81a511a2-de7e-c3c8-f641-3562c315ffa7
KIT=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit
export SARC_MOUNT_ROOT=$R ETVK_DEVICE_INDEX=0
gtemp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
# The card has dropped off the bus (Xid 79) while idle. If nvidia-smi fails: stop, never retry or reload.
gpu_alive() { [[ $(gtemp 2>/dev/null) =~ ^[0-9]+$ ]]; }
gpu_gone() { echo "GPU_GONE $(date -u +%FT%TZ): nvidia-smi failed ($1). Stopped; recovery needs a person." | tee -a $A/GPU_GONE; exit 70; }
# GPU users that are not ours: compute apps and graphics/Vulkan clients (pmon), plus known services by name.
others() {
  { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null
    nvidia-smi pmon -c 1 2>/dev/null | awk '$1 != "#" && $2 != "-" {print $2", "$NF}'
    pgrep -af 'llama-server|ComfyUI|comfyui|ollama|vllm|Runner.Worker' | grep -v pgrep | awk '{print $1", "$2}'
  } | grep -v 'llama_main\|test_llama_micr' | sort -u | tr ',' ':' | tr -d ' ' | tr '\n' ';'
}
# cool_start [max C] [timeout s]: wait until the GPU is at or below the given temperature.
cool_start() { local t0=$SECONDS; gpu_alive || gpu_gone cool_start
  while (( $(gtemp) > ${1:-50} && SECONDS - t0 < ${2:-300} )); do sleep 5; done; }
