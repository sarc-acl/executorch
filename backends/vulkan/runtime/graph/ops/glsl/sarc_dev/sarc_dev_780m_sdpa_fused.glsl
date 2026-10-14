/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, 780M (openspec/changes/sarc-1.5-780m-prefill-refine):
 * LLM-mode prefill SDPA in one kernel. One workgroup owns WG_TILE_M query rows
 * of one head and walks the context in blocks of WG_TILE_N columns, so the
 * S x context attention matrix is never written to memory:
 *
 *   pass A, per block: scores = Q K^T (fp32 accumulate, scaled, rounded to
 *           fp16 as sarc_sdpa_qk_coopmat does) -> running row maximum.
 *   pass B, per block: the same scores again -> e = exp(score - row max) as
 *           fp16, row sum of e in fp32 -> acc += e V (fp32 accumulate).
 *   end:    out = acc / row sum, rounded to fp16.
 *
 * Scores, row maxima and exp() arguments are the values the three-kernel path
 * computes. The difference is where the normalisation is rounded: that path
 * rounds e / sum to fp16 before attn*V; this kernel rounds e and divides the
 * fp32 accumulator. Not reachable without the hook in that change's hooks/.
 *
 * Fit (checked by impl/sarc_dev/Sdpa780mFused.cpp): fp16 buffers, head_dim ==
 * HEAD_DIM, S % WG_TILE_M == 0, input_pos % WG_TILE_N == 0.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_q", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_v", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}
// Output row stride (Q_H * head_dim): coopMatStore strides are never UBO-derived.
${layout_declare_spec_const(C, "int", "out_row_stride_arg", "0")}

const uint MMA = 16;
const uint HEAD_DIM = ${HEAD_DIM};
const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint WG_SIZE = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE;

// A subgroup owns SG_TILE_M rows and, of the scores, SG_TILE_C context columns;
// of the output, SG_TILE_D head_dim columns.
const uint MMAS_M = WG_TILE_M / SG_GRID_Y / MMA;
const uint MMAS_C = WG_TILE_N / SG_GRID_X / MMA;
const uint MMAS_D = HEAD_DIM / SG_GRID_X / MMA;

// Shared tiles, fp16 packed 8 per uvec4, rows padded by one uvec4.
const uint D_V8 = HEAD_DIM / 8u;
const uint D_STRIDE = D_V8 + 1u;
const uint P_STRIDE = WG_TILE_N / 8u + 1u;
shared uvec4 Qsh[WG_TILE_M * D_STRIDE];  // Q [s][d]
shared uvec4 KVsh[WG_TILE_N * D_STRIDE]; // K [c][d], then V [c][d]
shared uvec4 Psh[WG_TILE_M * P_STRIDE];  // scores, then e [s][c]

// Each invocation owns PART_W consecutive columns of one row of a block.
const uint PARTS = WG_SIZE / WG_TILE_M;
const uint PART_V8 = WG_TILE_N / PARTS / 8u;
shared float Rsh[WG_TILE_M * PARTS];   // per-invocation row maxima / sums
shared float Dsh[WG_TILE_M * MMA];     // row sums, one MMA tile wide

coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> sc[MMAS_M][MMAS_C];
coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> acc[MMAS_M][MMAS_D];

uvec4 pack8(const f16vec4 v0, const f16vec4 v1) {
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

void stage_k(const uint c_base, const uint row_stride, const uint head_base) {
  for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_N * D_V8; idx += WG_SIZE) {
    const uint lc = idx / D_V8;
    const uint d8 = idx % D_V8;
    const uint base = (c_base + lc) * row_stride + head_base + d8 * 2u;
    KVsh[lc * D_STRIDE + d8] = pack8(t_k[base], t_k[base + 1u]);
  }
}

void stage_v(const uint c_base, const uint row_stride, const uint head_base) {
  for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_N * D_V8; idx += WG_SIZE) {
    const uint lc = idx / D_V8;
    const uint d8 = idx % D_V8;
    const uint base = (c_base + lc) * row_stride + head_base + d8 * 2u;
    KVsh[lc * D_STRIDE + d8] = pack8(t_v[base], t_v[base + 1u]);
  }
}

// Scores of the staged block into Psh. K is staged [c][d], which is K^T read
// column-major: no transpose on write.
void qk_block(const uvec2 warp) {
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(0.0);
    }
  }
  [[unroll]] for (uint k = 0; k < HEAD_DIM / MMA; ++k) {
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      const uint row = MMA * (MMAS_M * warp.y + i);
      coopMatLoad(
          matA[i], Qsh, row * D_STRIDE + k * 2u, D_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      const uint col = MMA * (MMAS_C * warp.x + j);
      coopMatLoad(
          matB, KVsh, col * D_STRIDE + k * 2u, D_STRIDE,
          gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        sc[i][j] = coopMatMulAdd(matA[i], matB, sc[i][j]);
      }
    }
  }
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = sc[i][j] * inv_scale;
      const uint row = MMA * (MMAS_M * warp.y + i);
      const uint col = MMA * (MMAS_C * warp.x + j);
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(sc[i][j]),
          Psh, row * P_STRIDE + col / 8u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}

// acc += e V for the staged block.
void av_block(const uvec2 warp) {
  [[unroll]] for (uint k = 0; k < WG_TILE_N / MMA; ++k) {
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      const uint row = MMA * (MMAS_M * warp.y + i);
      coopMatLoad(
          matA[i], Psh, row * P_STRIDE + k * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      const uint col = MMA * (MMAS_D * warp.x + j);
      coopMatLoad(
          matB, KVsh, (MMA * k) * D_STRIDE + col / 8u, D_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        acc[i][j] = coopMatMulAdd(matA[i], matB, acc[i][j]);
      }
    }
  }
}

// Shared stores are ordered against coopMatLoad only with the explicit memory
// barrier (see the coopmat-lds-fence notes in sarc_sdpa_qk_coopmat.glsl).
#define SYNC memoryBarrierShared(); barrier()

void main() {
  const uvec2 warp = uvec2(gl_SubgroupID % SG_GRID_X, gl_SubgroupID / SG_GRID_X);
  const uint tid = gl_LocalInvocationID.x;
  const uint q_h = gl_WorkGroupID.z;

  // LLM layout: q_sizes WHCN {D, H_q, S, B}; k_sizes WHCN {D, H_kv, C_max, B}.
  const uint Q_H = uint(q_sizes.y);
  const uint S = uint(q_sizes.z);
  const uint KV_H = uint(k_sizes.y);
  const uint D4 = HEAD_DIM / 4u;
  const uint kv_h = KV_H < Q_H ? q_h / (Q_H / KV_H) : q_h;

  const uint s_base = WG_TILE_M * gl_WorkGroupID.y;
  if (s_base >= S) {
    return;
  }
  // Blocks past the one holding column s_base + WG_TILE_M - 1 + input_pos are
  // masked for every row of this tile.
  const uint context_len = uint(input_pos) + S;
  const uint num_blocks = min(
      context_len / WG_TILE_N,
      (s_base + WG_TILE_M - 1u + uint(input_pos)) / WG_TILE_N + 1u);

  const uint kv_row_stride = KV_H * D4;
  const uint kv_head_base = kv_h * D4;

  for (uint idx = tid; idx < WG_TILE_M * D_V8; idx += WG_SIZE) {
    const uint ls = idx / D_V8;
    const uint d8 = idx % D_V8;
    const uint base = ((s_base + ls) * Q_H + q_h) * D4 + d8 * 2u;
    Qsh[ls * D_STRIDE + d8] = pack8(t_q[base], t_q[base + 1u]);
  }

  // This invocation's elements of a block: row e_row, columns e_col + [0, 8 * PART_V8).
  const uint e_row = tid % WG_TILE_M;
  const uint e_part = tid / WG_TILE_M;
  const uint e_col = e_part * PART_V8 * 8u;
  const uint e_idx = e_row * P_STRIDE + e_col / 8u;
  // Column c of the row is visible when c <= e_last.
  const int e_last = int(s_base + e_row) + input_pos;
  const ivec4 LANE = ivec4(0, 1, 2, 3);

  // ---- pass A: row maxima ----
  float row_max = -1.0 / 0.0;
  for (uint b = 0; b < num_blocks; ++b) {
    stage_k(b * WG_TILE_N, kv_row_stride, kv_head_base);
    SYNC;
    qk_block(warp);
    SYNC;
    [[unroll]] for (uint i = 0; i < PART_V8; ++i) {
      const uvec4 u = Psh[e_idx + i];
      const int lim = e_last - int(b * WG_TILE_N + e_col + 8u * i);
      const vec4 lo = vec4(f16vec4(unpackFloat2x16(u.x), unpackFloat2x16(u.y)));
      const vec4 hi = vec4(f16vec4(unpackFloat2x16(u.z), unpackFloat2x16(u.w)));
      const vec4 mlo = mix(vec4(-1.0 / 0.0), lo, lessThanEqual(LANE, ivec4(lim)));
      const vec4 mhi = mix(vec4(-1.0 / 0.0), hi, lessThanEqual(LANE + 4, ivec4(lim)));
      const vec4 m4 = max(mlo, mhi);
      row_max = max(row_max, max(max(m4.x, m4.y), max(m4.z, m4.w)));
    }
  }
  Rsh[e_row * PARTS + e_part] = row_max;
  SYNC;
  [[unroll]] for (uint p = 0; p < PARTS; ++p) {
    row_max = max(row_max, Rsh[e_row * PARTS + p]);
  }

  // ---- pass B: e = exp(score - max), row sums, acc += e V ----
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      acc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(0.0);
    }
  }
  float row_sum = 0.0;
  for (uint b = 0; b < num_blocks; ++b) {
    stage_k(b * WG_TILE_N, kv_row_stride, kv_head_base);
    SYNC;
    qk_block(warp);
    SYNC;
    stage_v(b * WG_TILE_N, kv_row_stride, kv_head_base);
    [[unroll]] for (uint i = 0; i < PART_V8; ++i) {
      const uvec4 u = Psh[e_idx + i];
      const int lim = e_last - int(b * WG_TILE_N + e_col + 8u * i);
      const vec4 lo = vec4(f16vec4(unpackFloat2x16(u.x), unpackFloat2x16(u.y)));
      const vec4 hi = vec4(f16vec4(unpackFloat2x16(u.z), unpackFloat2x16(u.w)));
      const vec4 elo = mix(vec4(0.0), exp(lo - row_max), lessThanEqual(LANE, ivec4(lim)));
      const vec4 ehi = mix(vec4(0.0), exp(hi - row_max), lessThanEqual(LANE + 4, ivec4(lim)));
      row_sum += elo.x;
      row_sum += elo.y;
      row_sum += elo.z;
      row_sum += elo.w;
      row_sum += ehi.x;
      row_sum += ehi.y;
      row_sum += ehi.z;
      row_sum += ehi.w;
      Psh[e_idx + i] = pack8(f16vec4(elo), f16vec4(ehi));
    }
    SYNC;
    av_block(warp);
    SYNC;
  }

  // ---- normalise and store ----
  Rsh[e_row * PARTS + e_part] = row_sum;
  SYNC;
  row_sum = 0.0;
  [[unroll]] for (uint p = 0; p < PARTS; ++p) {
    row_sum += Rsh[e_row * PARTS + p];
  }
  for (uint j = e_part; j < MMA; j += PARTS) {
    Dsh[e_row * MMA + j] = row_sum;
  }
  SYNC;

  const uint out_row_stride = uint(out_row_stride_arg);
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    const uint row = MMA * (MMAS_M * warp.y + i);
    coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> den;
    coopMatLoad(den, Dsh, row * MMA, MMA, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      const uint col = MMA * (MMAS_D * warp.x + j);
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(acc[i][j] / den),
          t_output,
          (s_base + row) * out_row_stride + q_h * HEAD_DIM + col, out_row_stride,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
