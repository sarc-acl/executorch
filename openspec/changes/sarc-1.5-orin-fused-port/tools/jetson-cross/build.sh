#!/usr/bin/env bash
set -euo pipefail
cd /work/source/executorch
export PYTHONPATH=/work/source
common=(-DCMAKE_TOOLCHAIN_FILE=/recipe/aarch64.cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/venv/bin/python)
# orin-fused: an explicit shader compiler (container.sh, JETSON_CROSS_GLSLC); otherwise cmake finds the image's.
if [[ -n ${GLSLC:-} ]]; then "$GLSLC" --version | sed -n 1p; common+=(-DGLSLC_PATH="$GLSLC"); fi
mkdir -p /work/bundle
# Headers are architecture-neutral; do not add the host's /usr/include to ARM search paths.
aarch64-linux-gnu-g++ -O2 -std=c++17 /recipe/smoke.cpp \
  -I/work/source/executorch/backends/vulkan/third-party/Vulkan-Headers/include \
  -ldl -o /work/bundle/vulkan-smoke
# Orin campaign addition: vk-caps prints the subgroup sizes, compute limits and cooperative-matrix shapes.
aarch64-linux-gnu-g++ -O2 -std=c++17 /recipe/vk-caps.cpp \
  -I/work/source/executorch/backends/vulkan/third-party/Vulkan-Headers/include \
  -ldl -o /work/bundle/vk-caps || echo "vk-caps build failed (not fatal)"
if [[ ${1:-} == smoke ]]; then exit 0; fi
cmake --preset llm-release -B /work/build -G Ninja "${common[@]}" \
  -DCMAKE_INSTALL_PREFIX=/work/build \
  -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_ASR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF -DEXECUTORCH_BUILD_CUDA=OFF \
  -DEXECUTORCH_VULKAN_SHADER_COMPILE_NTHREADS=8
cmake --build /work/build --target install -j8
cmake -S examples/models/llama -B /work/build/examples/models/llama -G Ninja "${common[@]}" \
  -DCMAKE_PREFIX_PATH=/work/build -DCMAKE_FIND_ROOT_PATH='/work/build;/usr/aarch64-linux-gnu' \
  -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
cmake --build /work/build/examples/models/llama --target llama_main -j8
if [[ -d backends/vulkan/test/sarc_dev ]]; then
cmake -S backends/vulkan/test/sarc_dev -B /work/microbench -G Ninja "${common[@]}" \
  -DCMAKE_PREFIX_PATH=/work/build -DCMAKE_FIND_ROOT_PATH='/work/build;/usr/aarch64-linux-gnu' \
  -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_VULKAN_SHADER_COMPILE_NTHREADS=8
cmake --build /work/microbench --target test_llama_microbench -j8
fi
cp /work/build/examples/models/llama/llama_main /work/bundle/
cp /work/build/examples/models/llama/runner/libllama_runner.so /work/bundle/
[[ -f /work/microbench/test_llama_microbench ]] && cp /work/microbench/test_llama_microbench /work/bundle/ || true
file /work/bundle/*
