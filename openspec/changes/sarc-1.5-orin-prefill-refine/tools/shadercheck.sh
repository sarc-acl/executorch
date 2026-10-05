#!/bin/bash
# shadercheck.sh <name> <sarc_dev shader stem> [...]: workstation side. Compiles the named dev-zone shader
# templates of the WORKING COPY (before they are committed) with the cross image's glslc through the tree's
# gen_vulkan_spv.py, to find a syntax error before a build does. Not a build: nothing is linked and nothing of it
# is deployed. Under the desktop build lock. Output: <artifacts>/shadercheck/<name>/{out,log.txt}.
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
N=$1; shift; W=$A/shadercheck/$N; G=$ET/backends/vulkan/runtime/graph/ops/glsl; rm -rf $W; mkdir -p $W/src $W/out
cp $G/*.glslh $G/sarc/*.glslh $G/sarc_dev/*.glslh $W/src/ 2>/dev/null
for s in "$@"; do cp $G/sarc_dev/$s.glsl $G/sarc_dev/$s.yaml $W/src/ || exit 77; done
cp $ET/backends/vulkan/runtime/gen_vulkan_spv.py $W/
flock ~/.cache/gpu-lab/lock-desktop-build podman run --rm --userns=keep-id -v "$W:/work:Z" localhost/et-jetson-cross:jp7.2.1 \
  bash -c 'cd /work && python gen_vulkan_spv.py --glsl-path /work/src --output-path /work/out --glslc-path="$(which glslc)" --tmp-dir-path=/work/tmp --env VK_VERSION=1.1 --optimize 2>&1 | tail -40; echo "rc=${PIPESTATUS[0]}"; ls /work/out | grep -c "\.spv$"' > $W/log.txt 2>&1
tail -45 $W/log.txt
