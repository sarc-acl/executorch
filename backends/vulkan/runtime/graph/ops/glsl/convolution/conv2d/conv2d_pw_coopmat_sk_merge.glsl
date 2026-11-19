/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Merge of the split-K conv2d_pw_coopmat variants (_sk<n>): one invocation per
 * output texel (pixel m, 4-channel texel n4, n4 fastest) sums the K_SLICES
 * partial accumulator tiles from the scratch after the packed weights, in the
 * accumulator's type, adds the bias, applies the epilogue once and stores the
 * texel. Bindings, uniforms, push constants and specialization constants are
 * those of conv2d_pw_tiled, so the dispatch node can take the tiled kernel
 * instead on a resize to a shape the split-K variant cannot run.
 */

#version 450 core

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if ACC_TYPE == "f32":
  #define ACC_FP32
  #define VEC4_T vec4
$else:
  #define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_buffer(B, "r", "t_weight_u2", "uint", is_scalar_array=False, vec_size=2)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_residual", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_scale", DTYPE, "texture2d")}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
$if ACC_TYPE == "f32":
  float activation_a_f16;
  float activation_b_f16;
  float residual_alpha_f16;
  float activation_a;
  float activation_b;
  float residual_alpha;
$else:
  float activation_a;
  float activation_b;
  float residual_alpha;
  float activation_a_f32;
  float activation_b_f32;
  float residual_alpha_f32;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "stride_1_padding_0", "0")}
${layout_declare_spec_const(C, "bool", "has_activation", "false")}
${layout_declare_spec_const(C, "int", "activation_op", "0")}
${layout_declare_spec_const(C, "bool", "has_residual", "false")}
${layout_declare_spec_const(C, "bool", "has_scale", "false")}
${layout_declare_spec_const(C, "int", "residual_op", "0")}

#include "conv2d_epilogue.glslh"

const uint K_SLICES = ${K_SLICES};

void main() {
  const uint W_out = uint(out_sizes.x);
  const uint M = W_out * uint(out_sizes.y);
  const uint N4 = uint(div_up_4(out_sizes.z));
  const uint K4 = uint(div_up_4(in_sizes.z));
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= M * N4) {
    return;
  }
  const uint m = idx / N4;
  const uint n4 = idx % N4;
  const uint base = K4 * N4 * 4u;
  VEC4_T v = VEC4_T(0);
  [[unroll]] for (uint s = 0; s < K_SLICES; ++s) {
    const uint t = (s * M + m) * N4 + n4;
#ifdef ACC_FP32
    v += uintBitsToFloat(
        uvec4(t_weight_u2[base + 2u * t], t_weight_u2[base + 2u * t + 1u]));
#else
    const uvec2 u = t_weight_u2[base + t];
    v += VEC4_T(unpackFloat2x16(u.x), unpackFloat2x16(u.y));
#endif
  }
  v += VEC4_T(texelFetch(t_bias, ivec2(n4, 0), 0));
  const vec4 scale =
      has_scale ? texelFetch(t_scale, ivec2(n4, 0), 0) : vec4(1);
  const ivec3 pos = ivec3(m % W_out, m / W_out, n4);
  imageStore(t_out, pos, vec4(apply_epilogue(v, pos, scale)));
}
