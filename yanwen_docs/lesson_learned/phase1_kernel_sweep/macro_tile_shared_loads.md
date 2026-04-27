# Macro Tile Variants Need Strided Shared Loads

When the coopmat shader macro tile is no longer fixed at `64x64`, the original
one-thread-per-shared-load mapping is not valid for every variant. Smaller local
workgroups such as `16x16` or `32x32` have fewer invocations than the number of
`uvec4` loads needed for the A/B shared-memory tiles.

Use a strided loop over `gl_LocalInvocationID.x` for A and B shared-memory
loads:

```glsl
for (uint idx = gl_LocalInvocationID.x; idx < load_count; idx += INVOCATIONS) {
  ...
}
```

The hardware WMMA tile remains `16x16x16`; the macro-tile sweep only changes
how many of those HW tiles a workgroup covers.
