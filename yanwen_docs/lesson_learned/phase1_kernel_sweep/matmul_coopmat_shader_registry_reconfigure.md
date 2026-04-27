# Matmul coopmat shader registry reconfigure

## What was attempted

Added `matmul_coopmat.glsl` and `matmul_coopmat.yaml`, then rebuilt the runtime
and custom-op benchmark.

## What happened

The first `matmul_coopmat_bench` run aborted when the benchmark tried to create
the new shader:

```text
Could not find ShaderInfo with name matmul_coopmat_half
```

## Why it matters

Adding a new Vulkan GLSL/YAML shader is not enough if the top-level CMake
configure step is stale. The generated shader registry will not contain the new
entry, even if the custom-op target rebuilds.

## Workaround

Re-run the tested top-level configure command, then rebuild install and the
custom-op target:

```bash
cmake . -Bcmake-out-vk --preset "linux" -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_PAL_DEFAULT=posix -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-vk -j$(nproc) --target install --config Release
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc) --target matmul_coopmat_bench
```
