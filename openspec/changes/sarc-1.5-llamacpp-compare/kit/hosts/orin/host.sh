# host.sh adapter for the Jetson Orin Nano (the primary device), for kit/session.sh, which runs ON the device.
# Lock, sensors and device rules are those of the tuning campaign's tools/common.sh: gpu-thermal zone, devfreq
# clock of 17000000.gpu, VDD_IN power; no jetson_clocks, no nvpmodel, no sudo. The campaign's clock floors are
# per cell, at or just under the 612 MHz ceiling of this power mode; this adapter uses 590 MHz for every arm.
# The governor (nvhost_podgov) scales the clock with load, so an arm that loads the GPU less may sit lower:
# such runs are reported with their clock, as on the Radeon 780M.
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../common-simple.sh"
LOCK=b49259c9-868c-5b7c-b6f1-65a2bf4b63be
GPUDEV=/sys/class/devfreq/17000000.gpu; GPULOAD=/sys/devices/platform/17000000.gpu/load
for GZ in /sys/class/thermal/thermal_zone*; do [[ $(cat $GZ/type 2>/dev/null) == gpu-thermal ]] && break; GZ=""; done
for PWH in /sys/class/hwmon/hwmon*; do [[ $(cat $PWH/name 2>/dev/null) == ina3221 ]] && break; PWH=""; done
DEV_CLKMIN=590; DEV_BUSYMAX=""
export ETVK_DEVICE_INDEX=0
gtemp() { echo $(( $(<$GZ/temp) / 1000 )); }
# epoch_us clock_Hz busy power_uW temp_mC, every 50 ms (a tuned 1B prefill lasts 1.4 s here)
dev_sampler() { local f l t mv ma; while :; do
    read -r f < $GPUDEV/cur_freq; read -r l < $GPULOAD; read -r t < $GZ/temp; read -r mv < $PWH/in1_input; read -r ma < $PWH/curr1_input
    printf '%s %s %s %s %s\n' "${EPOCHREALTIME/./}" "$f" "$((l / 10))" "$((mv * ma))" "$t"; sleep 0.05; done > "$1" 2>/dev/null; }
