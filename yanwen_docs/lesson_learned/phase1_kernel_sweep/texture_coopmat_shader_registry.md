# New Vulkan Shader Files Need Reconfigure

Adding a new GLSL/YAML pair under `backends/vulkan/runtime/graph/ops/glsl/`
was not enough for the existing CMake build directory to include it in the
runtime shader registry. The build completed, but the benchmark aborted with:

```text
Could not find ShaderInfo with name linear_coopmat_texture3d_buffer
```

Rerun top-level CMake configure before rebuilding:

```bash
cmake -S . -B cmake-out-vk
cmake --build cmake-out-vk -j$(nproc) --target install --config Release
```

Also, during this run the top-level build failed because
`third-party/flatcc/lib` was missing, causing `ar` to fail while creating
`libflatccrt.a`. Recreating that directory with `mkdir -p third-party/flatcc/lib`
unblocked the build.
