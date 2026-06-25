#!/usr/bin/env bash
set -euo pipefail
cd /local/yanwen.xu/workspace/quant-dev/executorch
export ANDROID_NDK_HOME=/local/yanwen.xu/android-ndk-r29
export ANDROID_NDK=$ANDROID_NDK_HOME
GLSLC=/local/yanwen.xu/vulkan-sdk/1.4.350.1/x86_64/bin/glslc
source .venv/bin/activate
BD=cmake-out-android-vk-etdump
TC=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake

echo "########## STEP 1: core + vulkan + devtools/event-tracer ##########"
cmake . -B$BD --preset llm \
  -DCMAKE_TOOLCHAIN_FILE=$TC \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX=$BD -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_PAL_DEFAULT=posix -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DGLSLC_PATH=$GLSLC \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build $BD -j"$(nproc)" --target install --config Release

echo "########## STEP 2: llama_main runner (event tracer on) ##########"
cmake examples/models/llama -B$BD/examples/models/llama \
  -DCMAKE_TOOLCHAIN_FILE=$TC \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX=$BD -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_VULKAN=ON -DSUPPORT_REGEX_LOOKAHEAD=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DPYTHON_EXECUTABLE=python \
  -DCMAKE_CXX_FLAGS="-include algorithm -DET_EVENT_TRACER_ENABLED" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$(pwd)/$BD/lib -letdump -lflatccrt"
cmake --build $BD/examples/models/llama -j"$(nproc)" --config Release

echo "########## DONE ##########"
ls -la $BD/examples/models/llama/llama_main
strings -a $BD/examples/models/llama/llama_main | grep -c ET_EVENT_TRACER_ENABLED || true
