# host.sh: host constants of the Xe2 campaign (fedora-gpu-eval, card b70-0), sourced by every tool here.
# b70-0 is guest PCI 0000:01:00.0 = Vulkan device 0 (deviceUUID 868023e2-0000-0000-0100-000000000000).
XE2_ROOT=$HOME/hmz-sarc-xe2
A=${XE2_ARTIFACTS:-$XE2_ROOT/.artifacts}; ET=$XE2_ROOT/executorch
C=$ET/openspec/changes/sarc-1.5-xe2-prefill-refine
LOCK=868023e2-0000-0000-0100-000000000000
PCI=/sys/bus/pci/devices/0000:01:00.0
HW=$(echo $PCI/hwmon/hwmon*); FREQ=$PCI/tile0/gt0/freq0
export ETVK_DEVICE_INDEX=0 SARC_MOUNT_ROOT=$XE2_ROOT
gtemp_mc() { cat $HW/temp2_input; }   # package temperature, millidegrees C
# cool_start: wait until the package is within 3 C of the idle temperature recorded by the first session
# (A/idle_temp_mc), at most 5 min.
cool_start() { local lim=$(( $(cat $A/idle_temp_mc 2>/dev/null || echo 60000) + 3000 )) t0=$SECONDS
  while (( $(gtemp_mc) > lim && SECONDS - t0 < 300 )); do sleep 5; done; }
gpu_others() { pgrep -af 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|Runner.Worker|custom_ops' \
  | grep -v pgrep | awk '{print $1":"$2}' | tr '\n' ';'; }
TOOLS=$C/tools
