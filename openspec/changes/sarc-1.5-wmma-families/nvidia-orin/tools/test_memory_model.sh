#!/usr/bin/env bash
set -euo pipefail
out=$(realpath -m "${1:?supply an artifact output directory}")
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"
mkdir -p "$out"
exec 9>"$HOME/.cache/gpu-lab/lock-desktop-build"
flock 9
"${CXX:-c++}" -std=c++17 -O1 -ffunction-sections -fdata-sections \
  -I.. -Ibackends/vulkan/third-party/Vulkan-Headers/include \
  -Ibackends/vulkan/third-party/VulkanMemoryAllocator \
  openspec/changes/sarc-1.5-wmma-families/nvidia-orin/tools/memory_model_test.cpp \
  backends/vulkan/runtime/vk_api/Exception.cpp -Wl,--gc-sections \
  -o "$out/memory_model_test"
"$out/memory_model_test" | tee "$out/result.txt"
