# Texture Coopmat Needs Output Staging

The Vulkan cooperative matrix path can load/store cooperative matrices from
buffer-like memory or shared memory, but not directly from or to texture images.

For texture3D output, the working linear texture prototype stores the fp16
cooperative matrix result into a shared `uvec4` tile first, then normal shader
invocations unpack that shared tile and call `imageStore` on the texture3D
output.

This makes the texture path viable, but it adds overhead relative to the buffer
coopmat path. In the first benchmark, texture3D coopmat beat Stephen's texture
shader on eligible full-tile linear shapes, but remained slower than buffer
coopmat.
