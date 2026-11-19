#!/bin/bash
# device_info.sh DEVICE: writes OUT/DEVICE/device.txt (device, driver, Vulkan API version of Vulkan device 0, host
# OS, date, GPU clients) and the full raw query to OUT/DEVICE/vulkaninfo.txt (desktops: vulkaninfo; phones:
# cmd gpu vkjson). Read-only on the device.
set -u
. "$(dirname "$0")/devices.sh"; device_info "$1"; DEV=$1
mkdir -p "$OUT/$DEV"
if [ $KIND = adb ]; then
  rsh "cmd gpu vkjson" > "$OUT/$DEV/vulkaninfo.txt" 2>&1
  {
    echo "device: $DEV (adb $SERIAL), queried $(date -u +%FT%TZ)"
    echo "model: $(rsh 'getprop ro.product.model' | tr -d '\r')"
    echo "build: $(rsh 'getprop ro.build.fingerprint' | tr -d '\r')"
    "$OUT/../../main/executorch/.venv/bin/python" -I - "$OUT/$DEV/vulkaninfo.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["devices"][0]
p = d["properties"]
api = int(p["apiVersion"]); drv = int(p["driverVersion"])
print("gpu:", p["deviceName"])
print("vulkan api: %d.%d.%d" % (api >> 22, (api >> 12) & 0x3FF, api & 0xFFF))
print("driverVersion (raw):", drv, hex(drv))
dp = d.get("driverProperties") or d.get("VK_KHR_driver_properties", {}).get("driverPropertiesKHR", {})
for k in ("driverName", "driverInfo", "driverID"):
    if k in dp: print(k + ":", dp[k])
PY
  } > "$OUT/$DEV/device.txt"
else
  # B70 host has two identical cards: expose only card 0 (PCI 0000:01:00.0) to the Vulkan loader through Mesa's
  # device-select layer, so the probe never queries card 1, and refuse to record anything else. (The 2026-11-19
  # probe predates this and was unfiltered; see LOG.md section 8.)
  VKENV=
  if [ "$DEV" = b70 ]; then
    VKENV="DRI_PRIME=pci-0000_01_00_0 MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1"
    got=$(rsh "env $VKENV vulkaninfo --summary 2>/dev/null" | grep -E "^GPU[0-9]|deviceUUID" | tr -s ' \t\n' ' ' | sed 's/ $//')
    case "$got" in
      "GPU0: deviceUUID = 868023e2-0000-0000-0100-000000000000") ;;
      *) echo "b70: device-select did not restrict vulkaninfo to card 0: $got" >&2; exit 1;;
    esac
  fi
  rsh "env $VKENV vulkaninfo 2>/dev/null" > "$OUT/$DEV/vulkaninfo.txt"
  {
    echo "device: $DEV ($KIND${HOST:+ $HOST}), queried $(date -u +%FT%TZ)"
    echo "host: $(rsh 'uname -r; grep PRETTY /etc/os-release' | tr '\n' ' ')"
    rsh "env $VKENV vulkaninfo --summary 2>/dev/null" | awk '/^GPU0:/{f=1} /^GPU1:/{f=0} f' | grep -E "GPU0|apiVersion|driverVersion|vendorID|deviceID|deviceType|deviceName|driverID|driverName|driverInfo|deviceUUID|conformanceVersion"
    echo "gpu clients / lock at query time:"; "$(dirname "$0")/gpu_clients.sh" "$DEV" 2>&1 | sed 's/^/  /'
  } > "$OUT/$DEV/device.txt"
fi
cat "$OUT/$DEV/device.txt"
