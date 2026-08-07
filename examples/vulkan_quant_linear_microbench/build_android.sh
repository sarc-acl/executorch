#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

NDK_ROOT="${NDK_ROOT:-/local/yanwen.xu/android-ndk-r29}"
CLANGXX="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
API="${ANDROID_API:-28}"

python3 gen_spv_header.py shaders/linear_q4gsw_tiled_buffer_buffer_half.spv shaders/shader_q4gsw_spv.h shader_q4gsw_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_tiled_buffer_buffer_half.spv shaders/shader_dq8ca_spv.h shader_dq8ca_spv
python3 gen_spv_header.py shaders/linear_q4gsw_tiled_texture3d_texture2d_half.spv shaders/shader_q4gsw_texture_spv.h shader_q4gsw_texture_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half.spv shaders/shader_dq8ca_texture_spv.h shader_dq8ca_texture_spv
python3 gen_spv_header.py shaders/linear_q4gsw_coopmat_buffer_texture2d_half.spv shaders/shader_q4gsw_coopmat_texture_spv.h shader_q4gsw_coopmat_texture_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_coopmat_buffer_texture2d_half.spv shaders/shader_dq8ca_coopmat_texture_spv.h shader_dq8ca_coopmat_texture_spv
python3 gen_spv_header.py shaders/linear_q4gsw_coopmat_tsweep_t128x64k32g22s32_buffer_texture2d_half.spv shaders/shader_q4gsw_coopmat_tuned_spv.h shader_q4gsw_coopmat_tuned_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_coopmat_tsweep_t64x128k32g41s32_buffer_texture2d_half.spv shaders/shader_dq8ca_coopmat_tuned_spv.h shader_dq8ca_coopmat_tuned_spv
python3 gen_spv_header.py shaders/linear_dq8ca_q4gsw_coopmat_tsweep_t128x64k32g22s32_buffer_texture2d_half.spv shaders/shader_dq8ca_coopmat_sametile_spv.h shader_dq8ca_coopmat_sametile_spv

"$CLANGXX" --target=aarch64-linux-android"$API" -O2 -std=c++17 -static-libstdc++ microbench.cpp -lvulkan -o microbench_android

echo "Built ./microbench_android (aarch64, android-$API)"
