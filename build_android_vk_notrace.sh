#!/usr/bin/env bash
# Android arm64 cross-build of ExecuTorch Vulkan runtime + llama_main, no devtools/event tracer.
# Copied from release-1.3/executorch/build_android_vk_notrace.sh; build dir cmake-out-android.
set -euo pipefail
cd "$(dirname "$0")"
export ANDROID_NDK_HOME=/home/doremy/android-ndk-r29
export ANDROID_NDK="$ANDROID_NDK_HOME"
GLSLC=/home/doremy/vulkansdk/1.4.341.1/x86_64/bin/glslc
BUILD_DIR=cmake-out-android-mali
[[ "${1:-}" == "--clean" ]] && rm -rf "$BUILD_DIR"
source .venv/bin/activate
cmake . -B"$BUILD_DIR" --preset llm \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_PAL_DEFAULT=posix -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_DEVTOOLS=OFF -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
  -DGLSLC_PATH="$GLSLC" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build "$BUILD_DIR" -j"$(nproc)" --target install --config Release
cmake examples/models/llama -B"$BUILD_DIR/examples/models/llama" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_VULKAN=ON -DSUPPORT_REGEX_LOOKAHEAD=ON \
  -DPYTHON_EXECUTABLE=python \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build "$BUILD_DIR/examples/models/llama" -j"$(nproc)" --config Release
echo "spv shaders: $(find "$BUILD_DIR" -name "*.spv" | wc -l)"
file "$BUILD_DIR/examples/models/llama/llama_main"
