# env.sh: host constants of the RX 7600 campaign (host-ws1), sourced by every tool here.
A=<campaign-root>/.artifacts
ET=<campaign-root>/executorch
T=$ET/openspec/changes/sarc-1.5-rx7600-prefill-refine/tools
LOCK=rx7600-host
LOCKF=$HOME/.cache/gpu-lab/lock-$LOCK
MFLAT=$A/models                      # verify.sh --flat-models layout (links to the 2026-09-28 copies)
CARD=/sys/class/drm/card1/device
HW=/sys/class/hwmon/hwmon1           # amdgpu hwmon of card1 (0000:04:00.0)
[[ $(cat $HW/name 2>/dev/null) == amdgpu && $(readlink -f $HW/device) == $(readlink -f $CARD) ]] || {
  echo "env.sh: hwmon1 is not the amdgpu of card1" >&2; exit 70; }
export VK_ICD_FILENAMES=<mesa-install>/share/vulkan/icd.d/radeon_icd.x86_64.json
export ETVK_DEVICE_INDEX=0
export TMPDIR=$A/tmp
# big exports and builds of round 2 live on the local scratch disk (the root filesystem filled up on 2026-10-09 03:07 UTC); $A/{src,build}/rx7600/<tag> are symlinks to it
export SARC_BIG=<scratch>
gtemp() { local a b c; a=$(<$HW/temp1_input); b=$(<$HW/temp2_input); c=$(<$HW/temp3_input)
  echo $(( (a > b ? (a > c ? a : c) : (b > c ? b : c)) / 1000 )); }
[[ -e $A/GPU_GONE || -e $A/ABORTED ]] && { echo "env.sh: marker $(ls $A/GPU_GONE $A/ABORTED 2>/dev/null) present: stop" >&2; exit 71; }
true
