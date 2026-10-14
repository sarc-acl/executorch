/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, 780M (openspec/changes/sarc-1.5-780m-prefill-refine):
 * second structure of the fused prefill SDPA kernel. Same arithmetic as
 * sarc_dev_780m_sdpa_fused (two passes over the context per block of query
 * rows: row maxima, then e = exp(score - max) in fp16, row sums and
 * acc += e V in fp32, out = acc / sum); see that file for the description and
 * the fit conditions. What differs is how a workgroup is organised:
 *
 * - A subgroup owns SG_TILE_M whole rows (all WG_TILE_N columns of a block and
 *   all head_dim output columns), and its invocations own the elements of
 *   those rows. Scores, e, row maxima, row sums and the divisor are therefore
 *   written and read by one subgroup only and need no workgroup barrier.
 * - The K and V blocks are the only data shared per block. They are staged
 *   into two alternating slices, the next block prefetched into registers
 *   during the MMAs, so a block costs one barrier.
 * - V is staged transposed ([d][c]) and loaded column-major, as K is.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if ONE_PASS_WRONG:
  // MEASUREMENT ONLY, wrong results: no pass A (row maximum taken as 0).
  #define ONE_PASS_WRONG

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

// One column of subgroups: SG_GRID_X is 1.
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint WG_SIZE = SG_GRID_Y * SUBGROUP_SIZE;

const uint SG_TILE_M = WG_TILE_M / SG_GRID_Y;
const uint MMAS_M = SG_TILE_M / MMA;
const uint MMAS_C = WG_TILE_N / MMA;
const uint MMAS_D = HEAD_DIM / MMA;

// An invocation owns SEG_V8 uvec4 (8 fp16 each) of one row of a block.
const uint SEGS = SUBGROUP_SIZE / SG_TILE_M;
const uint SEG_V8 = WG_TILE_N / SEGS / 8u;

// Row padding of the shared tiles, in uvec4.
const uint PAD = ${PAD};
const uint D_V8 = HEAD_DIM / 8u;
const uint D_STRIDE = D_V8 + PAD;
const uint K_SLICE = WG_TILE_N * D_STRIDE;
// V^T [d][c] in uvec2 (4 fp16): a 4 x 4 block of V is four uvec2 stores.
const uint VT_STRIDE = WG_TILE_N / 4u + 2u * PAD;
const uint V_SLICE = HEAD_DIM * VT_STRIDE;
// A row of Psh also holds the 16 fp32 divisors of the row at the end.
const uint P_STRIDE = max(WG_TILE_N / 8u + PAD, 4u);

shared uvec4 Qsh[WG_TILE_M * D_STRIDE]; // Q [s][d]
shared uvec4 Ksh[2 * K_SLICE];          // K [c][d], two slices
shared uvec2 Vsh[2 * V_SLICE];          // V^T [d][c], two slices
shared uvec4 Psh[WG_TILE_M * P_STRIDE]; // scores, then e [s][c]; rows are subgroup-private
shared float Rsh[WG_TILE_M * SEGS];     // per-invocation row maxima / sums

const uint K_ITEMS = WG_TILE_N * D_V8;
const uint K_PASSES = (K_ITEMS + WG_SIZE - 1u) / WG_SIZE;
const uint V_ITEMS = (WG_TILE_N / 4u) * (HEAD_DIM / 4u);
const uint V_PASSES = (V_ITEMS + WG_SIZE - 1u) / WG_SIZE;

coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> sc[MMAS_M][MMAS_C];
coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> acc[MMAS_M][MMAS_D];

uvec4 pack8(const f16vec4 v0, const f16vec4 v1) {
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

// K item `item` of the block at context column c_base: 8 fp16 of one row.
uvec4 fetch_k(const uint item, const uint c_base, const uint row_stride, const uint head_base) {
  const uint base = (c_base + item / D_V8) * row_stride + head_base + (item % D_V8) * 2u;
  return pack8(t_k[base], t_k[base + 1u]);
}

void store_k(const uint slice, const uint item, const uvec4 v) {
  Ksh[slice * K_SLICE + (item / D_V8) * D_STRIDE + item % D_V8] = v;
}

// V item: rows c_base + 4 * (item / (HEAD_DIM / 4)) + [0, 4), one d4 texel each.
f16vec4 fetch_v(const uint item, const uint r, const uint c_base, const uint row_stride, const uint head_base) {
  const uint c = c_base + (item / (HEAD_DIM / 4u)) * 4u + r;
  return t_v[c * row_stride + head_base + item % (HEAD_DIM / 4u)];
}

void store_v(const uint slice, const uint item, const f16vec4 v[4]) {
  const uint c4 = item / (HEAD_DIM / 4u);
  const uint d = (item % (HEAD_DIM / 4u)) * 4u;
  [[unroll]] for (uint j = 0; j < 4u; ++j) {
    Vsh[slice * V_SLICE + (d + j) * VT_STRIDE + c4] = uvec2(
        packFloat2x16(f16vec2(v[0][j], v[1][j])),
        packFloat2x16(f16vec2(v[2][j], v[3][j])));
  }
}

// Scores of the block in K slice `slice` into this subgroup's rows of Psh.
// K is staged [c][d], which is K^T read column-major.
void qk_block(const uint slice, const uint row0) {
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(0.0);
    }
  }
  [[unroll]] for (uint k = 0; k < HEAD_DIM / MMA; ++k) {
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(
          matA[i], Qsh, (row0 + MMA * i) * D_STRIDE + k * 2u, D_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      coopMatLoad(
          matB, Ksh, slice * K_SLICE + (MMA * j) * D_STRIDE + k * 2u, D_STRIDE,
          gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        sc[i][j] = coopMatMulAdd(matA[i], matB, sc[i][j]);
      }
    }
  }
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = sc[i][j] * inv_scale;
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(sc[i][j]),
          Psh, (row0 + MMA * i) * P_STRIDE + j * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}

// acc += e V for the block in V slice `slice`.
void av_block(const uint slice, const uint row0) {
  [[unroll]] for (uint k = 0; k < WG_TILE_N / MMA; ++k) {
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(
          matA[i], Psh, (row0 + MMA * i) * P_STRIDE + k * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      coopMatLoad(
          matB, Vsh, slice * V_SLICE + (MMA * j) * VT_STRIDE + k * 4u, VT_STRIDE,
          gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        acc[i][j] = coopMatMulAdd(matA[i], matB, acc[i][j]);
      }
    }
  }
}

// Shared stores are ordered against coopMatLoad only with the explicit memory
// barrier (see the coopmat-lds-fence notes in sarc_sdpa_qk_coopmat.glsl).
// SYNC orders the workgroup; within a subgroup the memory barrier is enough.
#define SYNC memoryBarrierShared(); barrier()

void main() {
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

  // This subgroup's rows start at row0; this invocation owns row e_row,
  // columns e_col + [0, 8 * SEG_V8) of every block.
  const uint row0 = gl_SubgroupID * SG_TILE_M;
  const uint e_row = row0 + gl_SubgroupInvocationID % SG_TILE_M;
  const uint e_seg = gl_SubgroupInvocationID / SG_TILE_M;
  const uint e_col = e_seg * SEG_V8 * 8u;
  const uint e_idx = e_row * P_STRIDE + e_col / 8u;
  // Column c of the row is visible when c <= e_last.
  const int e_last = int(s_base + e_row) + input_pos;
  const ivec4 LANE = ivec4(0, 1, 2, 3);

  uvec4 next_k[K_PASSES];
  f16vec4 next_v[V_PASSES][4];

  // ---- pass A: row maxima ----
  float row_max = 0.0;
#ifndef ONE_PASS_WRONG
  row_max = -1.0 / 0.0;
  [[unroll]] for (uint p = 0; p < K_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    if (item < K_ITEMS) {
      store_k(0u, item, fetch_k(item, 0u, kv_row_stride, kv_head_base));
    }
  }
  for (uint b = 0; b < num_blocks; ++b) {
    SYNC;
    const bool more = b + 1u < num_blocks;
    if (more) {
      [[unroll]] for (uint p = 0; p < K_PASSES; ++p) {
        const uint item = p * WG_SIZE + tid;
        if (item < K_ITEMS) {
          next_k[p] = fetch_k(item, (b + 1u) * WG_TILE_N, kv_row_stride, kv_head_base);
        }
      }
    }
    qk_block(b & 1u, row0);
    memoryBarrierShared();
    [[unroll]] for (uint i = 0; i < SEG_V8; ++i) {
      const uvec4 u = Psh[e_idx + i];
      const int lim = e_last - int(b * WG_TILE_N + e_col + 8u * i);
      const vec4 lo = vec4(f16vec4(unpackFloat2x16(u.x), unpackFloat2x16(u.y)));
      const vec4 hi = vec4(f16vec4(unpackFloat2x16(u.z), unpackFloat2x16(u.w)));
      const vec4 mlo = mix(vec4(-1.0 / 0.0), lo, lessThanEqual(LANE, ivec4(lim)));
      const vec4 mhi = mix(vec4(-1.0 / 0.0), hi, lessThanEqual(LANE + 4, ivec4(lim)));
      const vec4 m4 = max(mlo, mhi);
      row_max = max(row_max, max(max(m4.x, m4.y), max(m4.z, m4.w)));
    }
    if (more) {
      [[unroll]] for (uint p = 0; p < K_PASSES; ++p) {
        const uint item = p * WG_SIZE + tid;
        if (item < K_ITEMS) {
          store_k((b + 1u) & 1u, item, next_k[p]);
        }
      }
    }
  }
  Rsh[e_row * SEGS + e_seg] = row_max;
  // Pass B restages slice 0, which the last block may still be reading.
  SYNC;
  [[unroll]] for (uint p = 0; p < SEGS; ++p) {
    row_max = max(row_max, Rsh[e_row * SEGS + p]);
  }
#endif

  // ---- pass B: e = exp(score - max), row sums, acc += e V ----
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      acc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(0.0);
    }
  }
  [[unroll]] for (uint p = 0; p < K_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    if (item < K_ITEMS) {
      store_k(0u, item, fetch_k(item, 0u, kv_row_stride, kv_head_base));
    }
  }
  [[unroll]] for (uint p = 0; p < V_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    if (item < V_ITEMS) {
      [[unroll]] for (uint r = 0; r < 4u; ++r) {
        next_v[p][r] = fetch_v(item, r, 0u, kv_row_stride, kv_head_base);
      }
      store_v(0u, item, next_v[p]);
    }
  }
  float row_sum = 0.0;
  for (uint b = 0; b < num_blocks; ++b) {
    SYNC;
    const bool more = b + 1u < num_blocks;
    if (more) {
      const uint c_next = (b + 1u) * WG_TILE_N;
      [[unroll]] for (uint p = 0; p < K_PASSES; ++p) {
        const uint item = p * WG_SIZE + tid;
        if (item < K_ITEMS) {
          next_k[p] = fetch_k(item, c_next, kv_row_stride, kv_head_base);
        }
      }
      [[unroll]] for (uint p = 0; p < V_PASSES; ++p) {
        const uint item = p * WG_SIZE + tid;
        if (item < V_ITEMS) {
          [[unroll]] for (uint r = 0; r < 4u; ++r) {
            next_v[p][r] = fetch_v(item, r, c_next, kv_row_stride, kv_head_base);
          }
        }
      }
    }
    qk_block(b & 1u, row0);
    memoryBarrierShared();
    [[unroll]] for (uint i = 0; i < SEG_V8; ++i) {
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
    memoryBarrierShared();
    av_block(b & 1u, row0);
    if (more) {
      [[unroll]] for (uint p = 0; p < K_PASSES; ++p) {
        const uint item = p * WG_SIZE + tid;
        if (item < K_ITEMS) {
          store_k((b + 1u) & 1u, item, next_k[p]);
        }
      }
      [[unroll]] for (uint p = 0; p < V_PASSES; ++p) {
        const uint item = p * WG_SIZE + tid;
        if (item < V_ITEMS) {
          store_v((b + 1u) & 1u, item, next_v[p]);
        }
      }
    }
  }

  // ---- normalise and store: all within the subgroup ----
  memoryBarrierShared();
  Rsh[e_row * SEGS + e_seg] = row_sum;
  memoryBarrierShared();
  row_sum = 0.0;
  [[unroll]] for (uint p = 0; p < SEGS; ++p) {
    row_sum += Rsh[e_row * SEGS + p];
  }
  for (uint j = e_seg; j < 4u; j += SEGS) {
    Psh[e_row * P_STRIDE + j] = uvec4(floatBitsToUint(row_sum));
  }
  memoryBarrierShared();

  const uint out_row_stride = uint(out_row_stride_arg);
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    const uint row = row0 + MMA * i;
    coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> den;
    coopMatLoad(den, Psh, row * P_STRIDE, P_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(acc[i][j] / den),
          t_output,
          (s_base + row) * out_row_stride + q_h * HEAD_DIM + MMA * j, out_row_stride,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
