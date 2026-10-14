# env.sh: constants of the RX 7900 XTX campaign, sourced by every tool here. It runs in two places:
#   control workstation (<campaign-root>/executorch/openspec/changes/<change>/tools): A = <campaign-root>/.artifacts,
#       host settings from the uncommitted $A/env.local (GPUHOST, GROOT, PY, GLSLC, ...)
#   GPU host (this directory rsynced to <gpu-root>/tools by sync-tools.sh, marker file ON_GPU_HOST):
#       A = <gpu-root>; sensors and the lock are looked up here
# Nothing host-specific is committed: names and absolute paths come from the directory this file is in and from env.local.
T=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
LOCK=7900xtx-gpu-host                # lock name on the GPU host (file name below; see env.local for the real one)
if [[ -e $T/ON_GPU_HOST ]]; then
  WHERE=gpu; A=$(dirname "$T"); ET=""
  [[ -f $A/env.gpu ]] && source "$A/env.gpu"           # LOCK=<real name>, written by sync-tools.sh from env.local
  export XDG_CACHE_HOME=${GCACHE:-$(dirname "$(dirname "$A")")/.cache}
  LOCKF=$XDG_CACHE_HOME/gpu-lab/lock-$LOCK
  MFLAT=$A/models                    # flat layout for verify.sh --flat-models (symlinks into the read-only model directory)
  CARD=/sys/class/drm/card1/device
  [[ $(cat $CARD/vendor 2>/dev/null) == 0x1002 && $(cat $CARD/device 2>/dev/null) == 0x744c ]] || {
    echo "env.sh: card1 is not the RX 7900 XTX (1002:744c)" >&2; exit 70; }
  HW=""; for h in /sys/class/hwmon/hwmon*; do
    [[ $(cat $h/name 2>/dev/null) == amdgpu && $(readlink -f $h/device) == $(readlink -f $CARD) ]] && HW=$h; done
  [[ -n $HW ]] || { echo "env.sh: no amdgpu hwmon of card1" >&2; exit 70; }
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json    # AMDVLK 2025.Q2.1 (LLPC); RADV only in a labelled session
  export ETVK_DEVICE_INDEX=0                                  # the only ICD loaded: the RX 7900 XTX is device 0
  export TMPDIR=$A/tmp; mkdir -p "$TMPDIR"
  gtemp() { local a b c; a=$(<$HW/temp1_input); b=$(<$HW/temp2_input); c=$(<$HW/temp3_input)
    echo $(( (a > b ? (a > c ? a : c) : (b > c ? b : c)) / 1000 )); }
  VERIFY=$A/sarc-tools/verify.sh     # the unmodified sarc/tools/verify.sh of the working copy, copied by sync-tools.sh (sha256 in sync.log)
  # cool-start waits (owner decision 2026-10-08 22:13 UTC): core temperatures only, edge (temp1) and junction (temp2); the memory sensor (temp3)
  # reads about 46 to 50 C at idle on this card and made every 48 C wait run its full cap. gtemp (max of all three) stays for the per-run cooling and the records.
  gtemp_core() { local a b; a=$(<$HW/temp1_input); b=$(<$HW/temp2_input); echo $(( (a > b ? a : b) / 1000 )); }
  export PATH=$(dirname "$A")/bin:$PATH     # glslc, spirv-dis, spirv-val of the GPU host's bin directory (not needed to run)
else
  WHERE=ws; ET=$(cd "$T/../../../.." && pwd); A=$(dirname "$ET")/.artifacts
  [[ -f $A/env.local ]] && source "$A/env.local"
  export TMPDIR=$A/tmp
  VERIFY=$ET/sarc/tools/verify.sh
fi
export A T
HOLDF=$A/HOLD
[[ -e $A/GPU_GONE || -e $A/ABORTED ]] && { echo "env.sh: marker $(ls $A/GPU_GONE $A/ABORTED 2>/dev/null) present: stop" >&2; exit 71; }
true
