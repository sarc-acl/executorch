#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Only needed if your system doesn't already have vulkan/vulkan.h on the
# default include path (e.g. from a `vulkan-headers`/Vulkan SDK package).
VULKAN_SDK_INC="${VULKAN_SDK_INC:-}"
INC_FLAG=""
[ -n "$VULKAN_SDK_INC" ] && INC_FLAG="-I$VULKAN_SDK_INC"

python3 gen_spv_header.py shaders/linear_q4gsw_tiled_buffer_buffer_half.spv shaders/shader_q4gsw_spv.h shader_q4gsw_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_tiled_buffer_buffer_half.spv shaders/shader_dq8ca_spv.h shader_dq8ca_spv
python3 gen_spv_header.py shaders/linear_q4gsw_tiled_texture3d_texture2d_half.spv shaders/shader_q4gsw_texture_spv.h shader_q4gsw_texture_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half.spv shaders/shader_dq8ca_texture_spv.h shader_dq8ca_texture_spv

g++ -O2 -std=c++17 $INC_FLAG microbench.cpp -lvulkan -o microbench

echo "Built ./microbench"
