/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Arc B580 (openspec/changes/sarc-1.5-b580-fused-port):
 * the fused prefill SDPA kernel of the 780M / RX 7600 campaigns
 * (sarc_dev_780m_sdpa_fused3sb.glsl on topic/rx7600-prefill-refine, the form
 * with a subgroupBarrier() after every memoryBarrierShared()) for the
 * cooperative-matrix shape ANV exposes: fp16 8 x 16 x 16, not 16 x 16 x 16.
 *
 * One kernel computes out = softmax(Q K^T / sqrt(d), causal) V for
 * WG_TILE_M whole query rows of one head: per block of WG_TILE_N context
 * columns, scores = Q K^T (fp32 accumulation, stored fp16), e =
 * exp(score - max) in fp16, row sums and acc += e V in fp32, out = acc / sum.
 *
 * A workgroup is SUBGROUPS subgroups of SUBGROUP_SIZE lanes (the pipeline
 * requires the subgroup size and the local size is SUBGROUPS times it;
 * impl/sarc_dev/b580/SdpaB580Fused.cpp). K and V are never staged:
 * - Q tiles are loaded once (AQ_REG: kept in registers) or for every product;
 * - K and V are read from the tile-packed copies sarc_dev_b580_sdpa_kvt
 *   writes before this kernel runs, in which every 16 x 16 operand tile is
 *   512 contiguous bytes (t_k [kv_h][c / 16][d / 16][c % 16][d % 16],
 *   t_vt [kv_h][c / 16][d][c % 16]);
 * - a block's scores and e, the row maxima, the row sums and the divisors
 *   live in shared memory. With one subgroup only that subgroup touches it
 *   and the barrier is subgroupBarrier(); with several, every subgroup
 *   writes its own score columns, every lane reads and rewrites its own
 *   segment of a row, every subgroup loads the whole block of e, and the
 *   barrier is barrier() (SYNC()).
 *
 * ONLINE: one pass instead of two. The row maximum is a running maximum:
 * each block's scores first update it; when it rises for a row of the
 * subgroup, that row's accumulator and sum are scaled by
 * exp(old max - new max) (the accumulator by an element-wise divide with a
 * divisor tile from shared memory) before the block's e, computed against the
 * new maximum, is added.
 *
 * What differs from the 780M kernel:
 * - MMA is MMA_M x MMA_N x MMA_K = 8 x 16 x 16: every tile with query rows
 *   (scores, e, accumulator, Q fragment, divisor) is 8 rows high, so
 *   MMAS_M = WG_TILE_M / 8 and the row offsets into Psh / Dsh / t_q / t_output
 *   step by 8. The operand tiles of K^T (d x c) and V (c x d) stay 16 x 16,
 *   so the packed copies have the 780M's layout.
 * - SUBGROUP_SIZE is 16. Lane = subgroup id * 16 + invocation id owns row
 *   lane % WG_TILE_M and segment lane / WG_TILE_M of it
 *   (SEGS = SUBGROUPS * 16 / WG_TILE_M segments a row).
 * - register file: a thread has 4096 bytes of registers for 16-lane values,
 *   and the fp32 accumulators of 8 rows x head_dim 128 are 4096 bytes. One
 *   subgroup owning whole rows therefore spills (the compiler reports 199 to
 *   1250 spilled values for head_dim 128) and is slower than the three
 *   kernels it replaces. MULTI_SG splits the accumulators' head_dim tiles and
 *   the score columns over the subgroups of a workgroup, which share the
 *   block's e through shared memory; QK_J_OUTER keeps one column of score
 *   tiles live; ACC_SH (accumulators in shared memory) was measured slower
 *   and is kept for the record. None of them changes a product or the order
 *   in which a tile accumulates; the row sum is added up per segment first.
 * - the rescale of the one-pass form has barriers, so with several subgroups
 *   the decision is taken for the workgroup: an atomic flag in shared memory
 *   (Gsh) instead of subgroupAny.
 * - only the packed form exists (no unpacked K / transposed V path, no
 *   measurement-only variant).
 * - the workgroup = SUBGROUPS full subgroups assumption is checked at run
 *   time (NaN output otherwise): on this driver nothing but the required
 *   subgroup size states it.
 *
 * Fit (SdpaB580Fused.cpp): fp16 buffers, head_dim == HEAD_DIM,
 * S % WG_TILE_M == 0, S % WG_TILE_N == 0, input_pos % WG_TILE_N == 0.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if AQ_REG:
  #define AQ_REG
$if ONLINE:
  #define ONLINE
$if QK_J_OUTER:
  // One score tile column at a time (needs AQ_REG): MMAS_M live score tiles instead of MMAS_M * MMAS_C.
  #define QK_J_OUTER
$if SUBGROUPS > 1:
  // A workgroup is SUBGROUPS subgroups that share the block's scores and e
  // through shared memory; each owns 1 / SUBGROUPS of the score tile columns
  // and of the head_dim tiles of the accumulators (needs QK_J_OUTER, not ACC_SH).
  #define MULTI_SG
$if ACC_SH:
  // The fp32 accumulators live in shared memory, loaded and stored around each block's products.
  #define ACC_SH

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
// coopMatLoad / coopMatStore strides are never UBO-derived: the row stride of
// q / out (Q_H * head_dim) and the context capacity of the packed copies.
${layout_declare_spec_const(C, "int", "out_row_stride_arg", "0")}
${layout_declare_spec_const(C, "int", "vt_stride_arg", "0")}

const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};
const uint HEAD_DIM = ${HEAD_DIM};
const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint SGS = ${SUBGROUPS};
const uint LANES = SUBGROUP_SIZE * SGS;

// Every read of a shared slot another invocation wrote follows one of these.
#ifdef MULTI_SG
#define SYNC() memoryBarrierShared(); barrier()
#define SG_ID gl_SubgroupID
#else
#define SYNC() memoryBarrierShared(); subgroupBarrier()
#define SG_ID 0u
#endif

// Query-row tiles are MMA_M high; context and head_dim tiles are 16 wide both
// as the N of one product and as the K of the other (MMA_N == MMA_K == 16).
const uint MMAS_M = WG_TILE_M / MMA_M;
const uint MMAS_C = WG_TILE_N / MMA_N;
const uint MMAS_D = HEAD_DIM / MMA_N;

// An invocation owns SEG_V8 uvec4 (8 fp16 each) of one row of a block.
const uint SEGS = LANES / WG_TILE_M;
const uint SEG_V8 = WG_TILE_N / SEGS / 8u;

// A row of Psh (padded by one uvec4) also holds the row's 16 fp32 divisors at the end.
const uint P_STRIDE = max(WG_TILE_N / 8u + 1u, 4u);
shared uvec4 Psh[WG_TILE_M * P_STRIDE]; // scores, then e [s][c]
shared float Rsh[WG_TILE_M * SEGS];     // per-invocation row maxima / sums
#ifdef ONLINE
shared vec4 Dsh[WG_TILE_M * 4u];        // per-row rescale divisors, one tile wide
#ifdef MULTI_SG
shared uint Gsh;                        // some row's maximum rose in this block
#endif
#endif

#ifdef ACC_SH
shared float Ash[WG_TILE_M * HEAD_DIM];  // accumulators [s][d]
#define ACC_TILE(i, j) Ash, MMA_M * (i) * HEAD_DIM + MMA_N * (j), HEAD_DIM, gl_CooperativeMatrixLayoutRowMajor
#else
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> acc[MMAS_M][MMAS_D / SGS];
#endif
#ifndef QK_J_OUTER
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> sc[MMAS_M][MMAS_C];
#endif
#ifdef AQ_REG
coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> aq[MMAS_M][MMAS_D];
#endif

uvec4 pack8(const f16vec4 v0, const f16vec4 v1) {
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

// Tile (context tile j, d slice k) of the block at k_base, and tile (d tile j,
// context tile k) of the block at vt_base; both 16 columns of 16.
#define K_TILE(k_base, j, k) (k_base) + ((j) * MMAS_D + (k)) * 256u, 16u
#define VT_TILE(vt_base, j, k) (vt_base) + ((k) * HEAD_DIM + 16u * (j)) * 16u, 16u

// Scores of the block at context column c0 into Psh. q_base is the element
// of row 0, d 0 of this tile in t_q; k_base that of the block in t_k.
void qk_block(const uint q_base, const uint k_base) {
#ifdef QK_J_OUTER
  // Same products in the same k order per tile as below.
  [[unroll]] for (uint jj = 0; jj < MMAS_C / SGS; ++jj) {
    const uint j = jj * SGS + SG_ID;
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> sc_j[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      sc_j[i] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
    [[unroll]] for (uint k = 0; k < MMAS_D; ++k) {
      coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      coopMatLoad(
          matB, t_k, K_TILE(k_base, j, k), gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
#ifdef AQ_REG
        sc_j[i] = coopMatMulAdd(aq[i][k], matB, sc_j[i]);
#else
        coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA;
        coopMatLoad(
            matA, t_q, q_base + MMA_M * i * uint(out_row_stride_arg) + MMA_K * k,
            uint(out_row_stride_arg), gl_CooperativeMatrixLayoutRowMajor);
        sc_j[i] = coopMatMulAdd(matA, matB, sc_j[i]);
#endif
      }
    }
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      sc_j[i] = sc_j[i] * inv_scale;
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(sc_j[i]),
          Psh, MMA_M * i * P_STRIDE + j * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#else
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_C; ++j) {
      sc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }
  [[unroll]] for (uint k = 0; k < MMAS_D; ++k) {
#ifndef AQ_REG
    coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(
          matA[i], t_q, q_base + MMA_M * i * uint(out_row_stride_arg) + MMA_K * k,
          uint(out_row_stride_arg), gl_CooperativeMatrixLayoutRowMajor);
    }
#endif
    coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
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
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(sc[i][j]),
          Psh, MMA_M * i * P_STRIDE + j * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#endif
}

// acc += e V for the block; vt_base is the element of the block in t_vt.
void av_block(const uint vt_base) {
#ifdef ACC_SH
  // Same products in the same k order per accumulator tile as below.
  coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matE[MMAS_M][MMAS_C];
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint k = 0; k < MMAS_C; ++k) {
      coopMatLoad(
          matE[i][k], Psh, MMA_M * i * P_STRIDE + k * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
  [[unroll]] for (uint j = 0; j < MMAS_D; ++j) {
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> acc_j[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(acc_j[i], ACC_TILE(i, j));
    }
    [[unroll]] for (uint k = 0; k < MMAS_C; ++k) {
      coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      coopMatLoad(
          matB, t_vt, VT_TILE(vt_base, j, k), gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        acc_j[i] = coopMatMulAdd(matE[i][k], matB, acc_j[i]);
      }
    }
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatStore(acc_j[i], ACC_TILE(i, j));
    }
  }
#else
  [[unroll]] for (uint k = 0; k < MMAS_C; ++k) {
    coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_M];
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
      coopMatLoad(
          matA[i], Psh, MMA_M * i * P_STRIDE + k * 2u, P_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
    [[unroll]] for (uint j = 0; j < MMAS_D / SGS; ++j) {
      coopMatLoad(
          matB, t_vt, VT_TILE(vt_base, SG_ID * (MMAS_D / SGS) + j, k),
          gl_CooperativeMatrixLayoutColumnMajor);
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        acc[i][j] = coopMatMulAdd(matA[i], matB, acc[i][j]);
      }
    }
  }
#endif
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
  // Blocks past the one holding column s_base + WG_TILE_M - 1 + input_pos are
  // masked for every row of this tile.
  const uint context_len = uint(input_pos) + S;
  const uint num_blocks = min(
      context_len / WG_TILE_N,
      (s_base + WG_TILE_M - 1u + uint(input_pos)) / WG_TILE_N + 1u);

  const uint q_base = (s_base * Q_H + q_h) * HEAD_DIM;
  // Everything below assumes the workgroup is exactly SGS subgroups of
  // SUBGROUP_SIZE: the pipeline requires SUBGROUP_SIZE and the local size is
  // SGS times it. If a driver ever splits the workgroup differently, the rows
  // are written as NaN (one writer per row) instead of being computed.
  if (gl_NumSubgroups != SGS || gl_SubgroupSize != SUBGROUP_SIZE) {
    if (gl_LocalInvocationIndex < WG_TILE_M) {
      const uint row = q_base + gl_LocalInvocationIndex * uint(out_row_stride_arg);
      for (uint d = 0; d < HEAD_DIM; ++d) {
        t_output[row + d] = float16_t(uintBitsToFloat(0x7fc00000u));
      }
    }
    return;
  }
  // Both copies hold vt_stride_arg / 16 context tiles of 16 * HEAD_DIM
  // elements per head; a block is WG_TILE_N / 16 of them.
  const uint kv_head = kv_h * uint(vt_stride_arg) * HEAD_DIM;
  const uint KV_BLOCK = WG_TILE_N * HEAD_DIM;

#ifdef AQ_REG
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint k = 0; k < MMAS_D; ++k) {
      coopMatLoad(
          aq[i][k], t_q, q_base + MMA_M * i * uint(out_row_stride_arg) + MMA_K * k,
          uint(out_row_stride_arg), gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#endif

  // This invocation owns row e_row, columns e_col + [0, 8 * SEG_V8) of every
  // block.
  const uint lane = SG_ID * SUBGROUP_SIZE + gl_SubgroupInvocationID;
  const uint e_row = lane % WG_TILE_M;
  const uint e_seg = lane / WG_TILE_M;
  const uint e_col = e_seg * SEG_V8 * 8u;
  const uint e_idx = e_row * P_STRIDE + e_col / 8u;
  // Column c of the row is visible when c <= e_last.
  const int e_last = int(s_base + e_row) + input_pos;
  const ivec4 LANE = ivec4(0, 1, 2, 3);

  // Every shared slot has exactly one writer per phase (a lane's own row and
  // segment, or one cooperative store of a tile only this subgroup owns); the
  // one exception is Gsh, which lanes only set with atomicOr and lane 0 alone
  // clears. Every read of a slot another invocation wrote, including every
  // coopMatLoad from shared memory, follows a SYNC() placed after that write.

  // ---- pass A: row maxima ----
  float row_max = -1.0 / 0.0;
#ifndef ONLINE
  for (uint b = 0; b < num_blocks; ++b) {
    qk_block(q_base, kv_head + b * KV_BLOCK);
    SYNC();
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
    SYNC();
  }
  Rsh[e_row * SEGS + e_seg] = row_max;
  SYNC();
  [[unroll]] for (uint p = 0; p < SEGS; ++p) {
    row_max = max(row_max, Rsh[e_row * SEGS + p]);
  }
  SYNC();
#endif

  // ---- pass B: e = exp(score - max), row sums, acc += e V ----
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_D / SGS; ++j) {
#ifdef ACC_SH
      coopMatStore(
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0),
          ACC_TILE(i, j));
#else
      acc[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
#endif
    }
  }
#if defined(ONLINE) && defined(MULTI_SG)
  if (lane == 0u) {
    Gsh = 0u;
  }
#endif
#if defined(ACC_SH) || defined(MULTI_SG)
  SYNC();
#endif
  float row_sum = 0.0;
  for (uint b = 0; b < num_blocks; ++b) {
    qk_block(q_base, kv_head + b * KV_BLOCK);
    SYNC();
#ifdef ONLINE
    // Running maximum of this row over the blocks so far, all segments.
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
      SYNC();
      [[unroll]] for (uint p = 0; p < SEGS; ++p) {
        blk_max = max(blk_max, Rsh[e_row * SEGS + p]);
      }
      SYNC();
    }
    const float new_max = max(row_max, blk_max);
#ifdef MULTI_SG
    // The rescale has barriers, so every subgroup must take it together.
    if (new_max > row_max) {
      atomicOr(Gsh, 1u);
    }
    SYNC();
    if (Gsh != 0u) {
#else
    if (subgroupAny(new_max > row_max)) {
#endif
      // exp(new - old) is 1 for a row whose maximum did not move and +inf
      // for the first visible block (old maximum -inf, accumulator 0).
      const float grow = new_max > row_max ? exp(new_max - row_max) : 1.0;
      row_sum = row_sum / grow;
      for (uint j = e_seg; j < 4u; j += SEGS) {
        Dsh[e_row * 4u + j] = vec4(grow);
      }
      SYNC();
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> grow_tile;
        coopMatLoad(grow_tile, Dsh, MMA_M * i * 4u, 4u, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint j = 0; j < MMAS_D / SGS; ++j) {
#ifdef ACC_SH
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> a;
          coopMatLoad(a, ACC_TILE(i, j));
          coopMatStore(a / grow_tile, ACC_TILE(i, j));
#else
          acc[i][j] = acc[i][j] / grow_tile;
#endif
        }
      }
#ifdef MULTI_SG
      if (lane == 0u) {
        Gsh = 0u;
      }
#endif
      SYNC();
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
    SYNC();
    av_block(kv_head + b * KV_BLOCK);
    SYNC();
  }

  // ---- normalise and store ----
  Rsh[e_row * SEGS + e_seg] = row_sum;
  SYNC();
  row_sum = 0.0;
  [[unroll]] for (uint p = 0; p < SEGS; ++p) {
    row_sum += Rsh[e_row * SEGS + p];
  }
  for (uint j = e_seg; j < 4u; j += SEGS) {
    Psh[e_row * P_STRIDE + j] = uvec4(floatBitsToUint(row_sum));
  }
  SYNC();

  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> den;
    coopMatLoad(den, Psh, MMA_M * i * P_STRIDE, P_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint j = 0; j < MMAS_D / SGS; ++j) {
#ifdef ACC_SH
      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> a;
      coopMatLoad(a, ACC_TILE(i, j));
#else
      const coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> a = acc[i][j];
#endif
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(a / den),
          t_output,
          q_base + MMA_M * i * uint(out_row_stride_arg) + MMA_N * (SG_ID * (MMAS_D / SGS) + j),
          uint(out_row_stride_arg),
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
