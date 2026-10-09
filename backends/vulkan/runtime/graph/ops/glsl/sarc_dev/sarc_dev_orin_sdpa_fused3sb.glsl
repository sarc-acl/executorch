/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Jetson Orin (openspec/changes/sarc-1.5-orin-fused-port): the fused prefill SDPA kernel.
 * The body, from `#version` on, is sarc_dev_780m_sdpa_fused3sb.glsl of topic/rx7600-prefill-refine (b3bb758e38)
 * plus one check (as the 4070 Ti port has it: the kernel writes nothing unless the workgroup is one full subgroup
 * of SUBGROUP_SIZE): the 780M's third structure (sarc_dev_780m_sdpa_fused3) with a subgroupBarrier() after every
 * memoryBarrierShared(). On this device (NVIDIA, subgroup size 32) the invocations of a subgroup are not
 * guaranteed to run in lockstep, so every read of a shared slot that another invocation wrote is ordered by that
 * execution barrier, and every shared slot has exactly one writing invocation (or is written by a cooperative
 * matrix store of the whole subgroup).
 *
 * A workgroup is one subgroup of 32 and owns WG_TILE_M (16 or 32) whole query rows of one head:
 * - Q tiles are loaded once (AQ_REG: kept in registers; else reloaded for every block);
 * - K is read column-major from the cache (K [c][d] is K^T), V column-major from t_vt, a transposed copy
 *   [kv_h][d][c] written by sarc_dev_orin_sdpa_vt before this kernel runs;
 * - PACKED: K and V are both read from tile-packed copies written by sarc_dev_orin_sdpa_kvt, in which every
 *   16 x 16 operand tile is 512 contiguous bytes (t_k [kv_h][c / 16][d / 16][c % 16][d % 16],
 *   t_vt [kv_h][c / 16][d][c % 16]);
 * - scores, e = exp(score - max), row maxima, row sums and the divisor live in shared memory that only this
 *   subgroup touches. Scores are accumulated in fp32 and stored fp16; e is fp16; acc += e V and the row sums are
 *   fp32; out = acc / sum, rounded once.
 *
 * Two passes over the context per block of rows (row maxima, then e, sums and acc), or ONLINE: one pass with a
 * running row maximum; when it rises for a row of the subgroup, that row's accumulator and sum are scaled by
 * exp(old max - new max) before the block's e, computed against the new maximum, is added.
 *
 * Fit (impl/sarc_dev/orin/SdpaOrinFused.cpp): fp16 buffers, head_dim == HEAD_DIM, S % WG_TILE_M == 0,
 * S % WG_TILE_N == 0, input_pos % WG_TILE_N == 0.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if AQ_REG:
  #define AQ_REG
$if PACKED:
  #define PACKED
$if ONLINE:
  #define ONLINE
$if ONE_PASS_WRONG:
  // MEASUREMENT ONLY, wrong results: no pass A (row maximum taken as 0).
  #define ONE_PASS_WRONG

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_q", DTYPE, IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, K_CACHE_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_vt", DTYPE, K_CACHE_STORAGE, is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}
// coopMatLoad / coopMatStore strides are never UBO-derived: the row strides of
// q / out (Q_H * head_dim), of the K cache (KV_H * head_dim) and of t_vt (its
// context capacity).
${layout_declare_spec_const(C, "int", "out_row_stride_arg", "0")}
${layout_declare_spec_const(C, "int", "kv_row_stride_arg", "0")}
${layout_declare_spec_const(C, "int", "vt_stride_arg", "0")}

const uint MMA = 16;
const uint HEAD_DIM = ${HEAD_DIM};
const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};

const uint MMAS_M = WG_TILE_M / MMA;
const uint MMAS_C = WG_TILE_N / MMA;
const uint MMAS_D = HEAD_DIM / MMA;

// An invocation owns SEG_V8 uvec4 (8 fp16 each) of one row of a block.
const uint SEGS = SUBGROUP_SIZE / WG_TILE_M;
const uint SEG_V8 = WG_TILE_N / SEGS / 8u;

// A row of Psh (padded by one uvec4) also holds the row's 16 fp32 divisors at the end.
const uint P_STRIDE = max(WG_TILE_N / 8u + 1u, 4u);
shared uvec4 Psh[WG_TILE_M * P_STRIDE]; // scores, then e [s][c]
shared float Rsh[WG_TILE_M * SEGS];     // per-invocation row maxima / sums
#ifdef ONLINE
shared vec4 Dsh[WG_TILE_M * 4u];        // per-row rescale divisors, one MMA tile wide
#endif

coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> sc[MMAS_M][MMAS_C];
coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> acc[MMAS_M][MMAS_D];
#ifdef AQ_REG
coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> aq[MMAS_M][MMAS_D];
#endif

uvec4 pack8(const f16vec4 v0, const f16vec4 v1) {
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

#ifdef PACKED
// Tile (context tile j, d slice k) of the block at k_base, and tile (d tile j,
// context tile k) of the block at vt_base.
#define K_TILE(k_base, j, k) (k_base) + ((j) * MMAS_D + (k)) * 256u, 16u
#define VT_TILE(vt_base, j, k) (vt_base) + ((k) * HEAD_DIM + MMA * (j)) * 16u, 16u
#else
#define K_TILE(k_base, j, k) (k_base) + MMA * (j) * uint(kv_row_stride_arg) + MMA * (k), uint(kv_row_stride_arg)
#define VT_TILE(vt_base, j, k) (vt_base) + MMA * (j) * uint(vt_stride_arg) + MMA * (k), uint(vt_stride_arg)
#endif

// Scores of the block at context column c0 into Psh. q_base is the element
// of row 0, d 0 of this tile in t_q; k_base that of column c0, d 0 in t_k.
void qk_block(const uint q_base, const uint k_base) {
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(0.0);
    }
  }
  [[unroll]] for (uint k = 0; k < MMAS_D; ++k) {
#ifndef AQ_REG
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(
          matA[i], t_q, q_base + MMA * i * uint(out_row_stride_arg) + MMA * k,
          uint(out_row_stride_arg), gl_CooperativeMatrixLayoutRowMajor);
    }
#endif
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      coopMatLoad(
          matB, t_k, K_TILE(k_base, j, k), gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
#ifdef AQ_REG
        sc[i][j] = coopMatMulAdd(aq[i][k], matB, sc[i][j]);
#else
        sc[i][j] = coopMatMulAdd(matA[i], matB, sc[i][j]);
#endif
      }
    }
  }
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = sc[i][j] * inv_scale;
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(sc[i][j]),
          Psh, MMA * i * P_STRIDE + j * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}

// acc += e V for the block; vt_base is the element of d 0, column c0 of this
// head in t_vt.
void av_block(const uint vt_base) {
  [[unroll]] for (uint k = 0; k < MMAS_C; ++k) {
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(
          matA[i], Psh, MMA * i * P_STRIDE + k * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      coopMatLoad(
          matB, t_vt, VT_TILE(vt_base, j, k), gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        acc[i][j] = coopMatMulAdd(matA[i], matB, acc[i][j]);
      }
    }
  }
}

void main() {
  const uint q_h = gl_WorkGroupID.z;

  // LLM layout: q_sizes WHCN {D, H_q, S, B}; k_sizes WHCN {D, H_kv, C_max, B}.
  const uint Q_H = uint(q_sizes.y);
  const uint S = uint(q_sizes.z);
  const uint KV_H = uint(k_sizes.y);
  const uint kv_h = KV_H < Q_H ? q_h / (Q_H / KV_H) : q_h;

  const uint s_base = WG_TILE_M * gl_WorkGroupID.y;
  if (s_base >= S) {
    return;
  }
  // orin: Psh, Rsh and Dsh are private to this subgroup, and every slot has
  // one writer, only if the workgroup is one full subgroup.
  if (gl_NumSubgroups != 1u || gl_SubgroupSize != SUBGROUP_SIZE) {
    return;
  }
  // Blocks past the one holding column s_base + WG_TILE_M - 1 + input_pos are
  // masked for every row of this tile.
  const uint context_len = uint(input_pos) + S;
  const uint num_blocks = min(
      context_len / WG_TILE_N,
      (s_base + WG_TILE_M - 1u + uint(input_pos)) / WG_TILE_N + 1u);

  const uint q_base = (s_base * Q_H + q_h) * HEAD_DIM;
#ifdef PACKED
  // Both copies hold vt_stride_arg / 16 context tiles of 16 * HEAD_DIM
  // elements per head; a block is WG_TILE_N / 16 of them.
  const uint k_head = kv_h * uint(vt_stride_arg) * HEAD_DIM;
  const uint vt_head = k_head;
  const uint K_BLOCK = WG_TILE_N * HEAD_DIM;
  const uint VT_BLOCK = WG_TILE_N * HEAD_DIM;
#else
  const uint k_head = kv_h * HEAD_DIM;
  const uint vt_head = kv_h * HEAD_DIM * uint(vt_stride_arg);
  const uint K_BLOCK = WG_TILE_N * uint(kv_row_stride_arg);
  const uint VT_BLOCK = WG_TILE_N;
#endif

#ifdef AQ_REG
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint k = 0; k < MMAS_D; ++k) {
      coopMatLoad(
          aq[i][k], t_q, q_base + MMA * i * uint(out_row_stride_arg) + MMA * k,
          uint(out_row_stride_arg), gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#endif

  // This invocation owns row e_row, columns e_col + [0, 8 * SEG_V8) of every
  // block.
  const uint e_row = gl_SubgroupInvocationID % WG_TILE_M;
  const uint e_seg = gl_SubgroupInvocationID / WG_TILE_M;
  const uint e_col = e_seg * SEG_V8 * 8u;
  const uint e_idx = e_row * P_STRIDE + e_col / 8u;
  // Column c of the row is visible when c <= e_last.
  const int e_last = int(s_base + e_row) + input_pos;
  const ivec4 LANE = ivec4(0, 1, 2, 3);

  // Shared stores are ordered against coopMatLoad only with the explicit
  // memory barrier (see the coopmat-lds-fence notes in
  // sarc_sdpa_qk_coopmat.glsl); everything shared here is subgroup-private.

  // ---- pass A: row maxima ----
  float row_max = 0.0;
#ifdef ONLINE
  row_max = -1.0 / 0.0;
#elif !defined(ONE_PASS_WRONG)
  row_max = -1.0 / 0.0;
  for (uint b = 0; b < num_blocks; ++b) {
    qk_block(q_base, b * K_BLOCK + k_head);
    memoryBarrierShared();
    subgroupBarrier();
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
    memoryBarrierShared();
    subgroupBarrier();
  }
  Rsh[e_row * SEGS + e_seg] = row_max;
  memoryBarrierShared();
  subgroupBarrier();
  [[unroll]] for (uint p = 0; p < SEGS; ++p) {
    row_max = max(row_max, Rsh[e_row * SEGS + p]);
  }
  memoryBarrierShared();
  subgroupBarrier();
#endif

  // ---- pass B: e = exp(score - max), row sums, acc += e V ----
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      acc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(0.0);
    }
  }
  float row_sum = 0.0;
  for (uint b = 0; b < num_blocks; ++b) {
    qk_block(q_base, b * K_BLOCK + k_head);
    memoryBarrierShared();
    subgroupBarrier();
#ifdef ONLINE
    // Running maximum of this row over the blocks so far, both segments.
    float blk_max = -1.0 / 0.0;
    uvec4 own[SEG_V8];
    [[unroll]] for (uint i = 0; i < SEG_V8; ++i) {
      const uvec4 u = Psh[e_idx + i];
      own[i] = u;
      const int lim = e_last - int(b * WG_TILE_N + e_col + 8u * i);
      const vec4 lo = vec4(f16vec4(unpackFloat2x16(u.x), unpackFloat2x16(u.y)));
      const vec4 hi = vec4(f16vec4(unpackFloat2x16(u.z), unpackFloat2x16(u.w)));
      const vec4 mlo = mix(vec4(-1.0 / 0.0), lo, lessThanEqual(LANE, ivec4(lim)));
      const vec4 mhi = mix(vec4(-1.0 / 0.0), hi, lessThanEqual(LANE + 4, ivec4(lim)));
      const vec4 m4 = max(mlo, mhi);
      blk_max = max(blk_max, max(max(m4.x, m4.y), max(m4.z, m4.w)));
    }
    if (SEGS > 1u) {
      Rsh[e_row * SEGS + e_seg] = blk_max;
      memoryBarrierShared();
      subgroupBarrier();
      [[unroll]] for (uint p = 0; p < SEGS; ++p) {
        blk_max = max(blk_max, Rsh[e_row * SEGS + p]);
      }
      memoryBarrierShared();
      subgroupBarrier();
    }
    const float new_max = max(row_max, blk_max);
    if (subgroupAny(new_max > row_max)) {
      // exp(new - old) is 1 for a row whose maximum did not move and +inf
      // for the first visible block (old maximum -inf, accumulator 0).
      const float grow = new_max > row_max ? exp(new_max - row_max) : 1.0;
      row_sum = row_sum / grow;
      for (uint j = e_seg; j < 4u; j += SEGS) {
        Dsh[e_row * 4u + j] = vec4(grow);
      }
      memoryBarrierShared();
      subgroupBarrier();
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> grow_tile;
        coopMatLoad(grow_tile, Dsh, MMA * i * 4u, 4u, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
          acc[i][j] = acc[i][j] / grow_tile;
        }
      }
      memoryBarrierShared();
      subgroupBarrier();
      row_max = new_max;
    }
#endif
    [[unroll]] for (uint i = 0; i < SEG_V8; ++i) {
#ifdef ONLINE
      const uvec4 u = own[i];
#else
      const uvec4 u = Psh[e_idx + i];
#endif
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
    subgroupBarrier();
    av_block(vt_head + b * VT_BLOCK);
    memoryBarrierShared();
    subgroupBarrier();
  }

  // ---- normalise and store ----
  Rsh[e_row * SEGS + e_seg] = row_sum;
  memoryBarrierShared();
  subgroupBarrier();
  row_sum = 0.0;
  [[unroll]] for (uint p = 0; p < SEGS; ++p) {
    row_sum += Rsh[e_row * SEGS + p];
  }
  for (uint j = e_seg; j < 4u; j += SEGS) {
    Psh[e_row * P_STRIDE + j] = uvec4(floatBitsToUint(row_sum));
  }
  memoryBarrierShared();
  subgroupBarrier();

  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    coopmat<float, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator> den;
    coopMatLoad(den, Psh, MMA * i * P_STRIDE, P_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA, MMA, gl_MatrixUseAccumulator>(acc[i][j] / den),
          t_output,
          q_base + MMA * i * uint(out_row_stride_arg) + MMA * j, uint(out_row_stride_arg),
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
