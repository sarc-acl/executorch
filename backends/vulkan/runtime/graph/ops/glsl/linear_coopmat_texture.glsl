/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix linear shader for texture3d input/output and buffer
 * packed weights. This is an experimental fp16-only path for comparing against
 * Stephen's texture linear_vec shader without changing graph-wide storage.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

${define_required_extensions("texture3d", "half")}
${define_required_extensions("buffer", "half")}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", "half", "texture3d", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", "half", "texture3d", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_packed", "half", "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const uint lM = 16;
const uint lN = 16;
const uint lK = 16;
const uint TILE_M = 64;
const uint TILE_N = 64;
const uint TILE_K = 32;

const uint WG_WIDTH = 2;
const uint WG_HEIGHT = 2;
const uint NUM_SUBGROUPS = 4;
const uint INVOCATIONS = 64 * NUM_SUBGROUPS;

const uint C_ROWS = TILE_M / WG_HEIGHT / lM;
const uint C_COLS = TILE_N / WG_WIDTH / lN;

const uint FP16_PER_VEC4 = 8;

const uint A_STRIDE_VEC4 = TILE_K / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = TILE_N / FP16_PER_VEC4;
const uint C_STRIDE_VEC4 = TILE_N / FP16_PER_VEC4;

shared uvec4 Ash[TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[TILE_K * B_STRIDE_VEC4];
shared uvec4 Csh[TILE_M * C_STRIDE_VEC4];

coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];

vec4 unpack_f16x4(uint lo, uint hi) {
  vec2 xy = unpackHalf2x16(lo);
  vec2 zw = unpackHalf2x16(hi);
  return vec4(xy, zw);
}

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uint batch = gl_WorkGroupID.z;
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % WG_WIDTH,
      gl_SubgroupID / WG_WIDTH);

  const uint K = uint(mat1_sizes.x);
  const uint M = uint(mat1_sizes.y);
  const uint N = uint(out_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (N + 3u) / 4u;

  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      result[i][j] =
          coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);
    }
  }

  const uint INVS_PER_ROW_A = TILE_K / FP16_PER_VEC4;
  const uint INVS_PER_ROW_B = TILE_N / FP16_PER_VEC4;

  const uint a_row_base = TILE_M * tileID.y;
  const uint b_col_base = TILE_N * tileID.x;

  for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {
    for (uint idx = gl_LocalInvocationID.x;
         idx < TILE_M * INVS_PER_ROW_A;
         idx += INVOCATIONS) {
      uint a_col = idx % INVS_PER_ROW_A;
      uint a_row_offset = idx / INVS_PER_ROW_A;
      uint row = a_row_base + a_row_offset;
      uint k_elem = chunkK + a_col * FP16_PER_VEC4;
      uint k_hv4 = k_elem / 4u;

      vec4 v0 = texelFetch(t_mat1, ivec3(k_hv4, row, batch), 0);
      vec4 v1 = texelFetch(t_mat1, ivec3(k_hv4 + 1u, row, batch), 0);
      Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
          packHalf2x16(vec2(v0.xy)), packHalf2x16(vec2(v0.zw)),
          packHalf2x16(vec2(v1.xy)), packHalf2x16(vec2(v1.zw)));
    }

    for (uint idx = gl_LocalInvocationID.x;
         idx < TILE_K * INVS_PER_ROW_B;
         idx += INVOCATIONS) {
      uint b_col = idx % INVS_PER_ROW_B;
      uint b_row_offset = idx / INVS_PER_ROW_B;
      uint k_row = chunkK + b_row_offset;
      uint k4 = k_row >> 2u;
      uint dk = k_row & 3u;
      uint n_elem = b_col_base + b_col * FP16_PER_VEC4;
      uint n4_0 = n_elem >> 2u;

      f16vec4 v0 = t_weight_packed[(k4 * N4 + n4_0) * 4u + dk];
      f16vec4 v1 = t_weight_packed[(k4 * N4 + n4_0 + 1u) * 4u + dk];
      Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
          packHalf2x16(vec2(v0.xy)), packHalf2x16(vec2(v0.zw)),
          packHalf2x16(vec2(v1.xy)), packHalf2x16(vec2(v1.zw)));
    }

    barrier();

    [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k) {
      uint k_start = lK * k;

      coopmat<float16_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
      [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        uint row_a = lM * (C_ROWS * warpInTile.y + i);
        coopMatLoad(
            matA[i],
            Ash,
            row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
            A_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);
      }

      coopmat<float16_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
      [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
        uint col_b = lN * (C_COLS * warpInTile.x + j) / FP16_PER_VEC4;
        coopMatLoad(
            matB,
            Bsh,
            k_start * B_STRIDE_VEC4 + col_b,
            B_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
          result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
        }
      }
    }

    barrier();
  }

  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      uint gi = TILE_M * tileID.y + lM * (C_ROWS * warpInTile.y + i);
      uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);
      coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(
              result[i][j]);
      coopMatStore(
          out_tile,
          Csh,
          (gi - a_row_base) * C_STRIDE_VEC4 + (gj - b_col_base) / FP16_PER_VEC4,
          C_STRIDE_VEC4,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }

  barrier();

  for (uint idx = gl_LocalInvocationID.x;
       idx < TILE_M * C_STRIDE_VEC4;
       idx += INVOCATIONS) {
    uint local_m = idx / C_STRIDE_VEC4;
    uint local_n8 = idx % C_STRIDE_VEC4;
    uint m = a_row_base + local_m;
    uint n4 = (b_col_base / 4u) + local_n8 * 2u;

    if (m < M && n4 < N4) {
      uvec4 packed = Csh[idx];
      imageStore(
          t_output,
          ivec3(n4, m, batch),
          unpack_f16x4(packed.x, packed.y));
      if (n4 + 1u < N4) {
        imageStore(
            t_output,
            ivec3(n4 + 1u, m, batch),
            unpack_f16x4(packed.z, packed.w));
      }
    }
  }
}
