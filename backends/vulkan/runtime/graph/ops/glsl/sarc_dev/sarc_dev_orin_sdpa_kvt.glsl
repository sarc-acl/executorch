/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Jetson Orin (openspec/changes/sarc-1.5-orin-fused-port; the 780M's
 * sarc_dev_780m_sdpa_kvt.glsl, unchanged from `#version` on):
 * writes the tile-packed copies of the K and V caches [c][kv_h][d] that the
 * PACKED variants of sarc_dev_orin_sdpa_fused3sb load, for c < context_len:
 *   t_kt [kv_h][c / 16][d / 16][c % 16][d % 16]   (a K^T tile: 16 runs of 16 d)
 *   t_vt [kv_h][c / 16][d][c % 16]                (a V tile: 16 runs of 16 c)
 * One invocation moves a 4 x 4 block of each. Dispatch:
 * (context_len / 4, D / 4, KV_H).
 */

#version 450 core

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_kt", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "w", "t_vt", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_v", DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "v_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Context capacity of the copies (a multiple of 16).
${layout_declare_spec_const(C, "int", "vt_stride_arg", "0")}

void main() {
  // v_sizes WHCN {D, H_kv, C_max, B}; q_sizes WHCN {D, H_q, S, B}.
  const uint D = uint(v_sizes.x);
  const uint KV_H = uint(v_sizes.y);
  const uint context_len = uint(input_pos) + uint(q_sizes.z);
  const uint c4 = gl_GlobalInvocationID.x;
  const uint d4 = gl_GlobalInvocationID.y;
  const uint kv_h = gl_GlobalInvocationID.z;
  if (c4 * 4u >= context_len || d4 * 4u >= D || kv_h >= KV_H) {
    return;
  }
  const uint c = c4 * 4u;
  const uint d = d4 * 4u;
  // First element of this head's context tile c / 16 (16 * D elements a tile).
  const uint tile = (kv_h * (uint(vt_stride_arg) / 16u) + c / 16u) * 16u * D;
  f16vec4 v[4];
  [[unroll]] for (uint r = 0; r < 4u; ++r) {
    const uint src = ((c + r) * KV_H + kv_h) * (D / 4u) + d4;
    v[r] = t_v[src];
    t_kt[(tile + (d / 16u) * 256u + (c % 16u + r) * 16u + d % 16u) / 4u] = t_k[src];
  }
  [[unroll]] for (uint j = 0; j < 4u; ++j) {
    t_vt[(tile + (d + j) * 16u + c % 16u) / 4u] =
        f16vec4(v[0][j], v[1][j], v[2][j], v[3][j]);
  }
}
