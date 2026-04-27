/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix matmul following matmul_vec conventions.
 *
 * Computes: D = A * B, where A is [M, K], B is [K, N], and D is [M, N].
 * Current benchmark variant is buffer-only and assumes M/N/K are compatible
 * with the 64x64x32 software tile.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if DTYPE == "half":
  #define IS_FP16_INPUT
$if DTYPE == "float":
  #define IS_FP32_INPUT

$if ACCUMULATOR_TYPE == "half":
  #define USE_FP16_ACCUMULATOR

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat2", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "mat2_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const uint lM = 16;
const uint lN = 16;
const uint lK = 16;
const uint TILE_M = ${TILE_M};
const uint TILE_N = ${TILE_N};
const uint TILE_K = ${TILE_K};

const uint WG_WIDTH = ${WG_WIDTH};
const uint WG_HEIGHT = ${WG_HEIGHT};
const uint NUM_SUBGROUPS = ${NUM_SUBGROUPS};
const uint INVOCATIONS = 64 * NUM_SUBGROUPS;

const uint C_ROWS = TILE_M / WG_HEIGHT / lM;
const uint C_COLS = TILE_N / WG_WIDTH / lN;

const uint FP16_PER_VEC4 = 8;

const uint A_STRIDE_VEC4 = (TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;

shared uvec4 Ash[TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[TILE_K * B_STRIDE_VEC4];

#ifdef USE_FP16_ACCUMULATOR
coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];
#else
coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];
#endif

#ifdef IS_FP32_INPUT
uvec2 f32x4_to_f16x4(vec4 v) {
  return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#endif

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % WG_WIDTH,
      gl_SubgroupID / WG_WIDTH);

  const uint K = uint(mat1_sizes.x);
  const uint N = uint(mat2_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (N + 3u) / 4u;

  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
#ifdef USE_FP16_ACCUMULATOR
      result[i][j] =
          coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(float16_t(0.0));
#else
      result[i][j] =
          coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);
#endif
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

#ifdef IS_FP16_INPUT
      uint k_hv4 = k_elem / 4u;
      f16vec4 v0 = t_mat1[row * K4 + k_hv4];
      f16vec4 v1 = t_mat1[row * K4 + k_hv4 + 1u];
      Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
          packHalf2x16(vec2(v0.xy)), packHalf2x16(vec2(v0.zw)),
          packHalf2x16(vec2(v1.xy)), packHalf2x16(vec2(v1.zw)));
#else
      uint k_vec4 = k_elem / 4u;
      vec4 v0 = t_mat1[row * K4 + k_vec4];
      vec4 v1 = t_mat1[row * K4 + k_vec4 + 1u];
      uvec2 h0 = f32x4_to_f16x4(v0);
      uvec2 h1 = f32x4_to_f16x4(v1);
      Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(h0, h1);
#endif
    }

    for (uint idx = gl_LocalInvocationID.x;
         idx < TILE_K * INVS_PER_ROW_B;
         idx += INVOCATIONS) {
      uint b_col = idx % INVS_PER_ROW_B;
      uint b_row_offset = idx / INVS_PER_ROW_B;
      uint row = chunkK + b_row_offset;
      uint n_elem = b_col_base + b_col * FP16_PER_VEC4;

#ifdef IS_FP16_INPUT
      uint n_hv4 = n_elem / 4u;
      f16vec4 v0 = t_mat2[row * N4 + n_hv4];
      f16vec4 v1 = t_mat2[row * N4 + n_hv4 + 1u];
      Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
          packHalf2x16(vec2(v0.xy)), packHalf2x16(vec2(v0.zw)),
          packHalf2x16(vec2(v1.xy)), packHalf2x16(vec2(v1.zw)));
#else
      uint n_vec4 = n_elem / 4u;
      vec4 v0 = t_mat2[row * N4 + n_vec4];
      vec4 v1 = t_mat2[row * N4 + n_vec4 + 1u];
      uvec2 h0 = f32x4_to_f16x4(v0);
      uvec2 h1 = f32x4_to_f16x4(v1);
      Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(h0, h1);
#endif
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
#ifdef IS_FP16_INPUT
#ifdef USE_FP16_ACCUMULATOR
      coopMatStore(
          result[i][j],
          t_output,
          gi * N + gj,
          N,
          gl_CooperativeMatrixLayoutRowMajor);
#else
      coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(
              result[i][j]);
      coopMatStore(
          out_tile,
          t_output,
          gi * N + gj,
          N,
          gl_CooperativeMatrixLayoutRowMajor);
#endif
#else
      coopMatStore(
          result[i][j],
          t_output,
          gi * N + gj,
          N,
          gl_CooperativeMatrixLayoutRowMajor);
#endif
    }
  }
}
