/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix SDPA QK^T kernel (prefill / LLM mode).
 *
 * Computes per head q_h:  attn[s, c] = inv_scale * sum_d Q[s, d] * K[c, d]
 *   Q = q_projected, DHSB [S, Q_H, D]   index (vec4 along d): (s*Q_H + q_h)*D4 + d4
 *   K = k_cache,     DHSB [context_len, KV_H, D]  index: (c*KV_H + kv_h)*D4 + d4
 *       (GQA: kv_h = q_h / (Q_H/KV_H))
 *   attn = attn_weights, head-contiguous [S_aligned, C4*4]
 *       scalar index: (q_h*S_aligned + s)*(C4*4) + c
 *   then the causal mask sets attn[s,c] = -inf where c > s + input_pos.
 *
 * Reduction dim is D (head_dim) -> num_k_chunks = D / WG_TILE_K (2 or 4).
 *
 * Two structural differences from coopmat_mm.glsl:
 *  1. K is consumed transposed (we need Q*K^T). Rather than a ColumnMajor load
 *     of packed shared memory, K is staged TRANSPOSED into an fp16 shared array
 *     laid out [d][c] (scatter on write, since native K has d contiguous), so
 *     the MMA loop reads it RowMajor exactly like A. Q is likewise staged into
 *     an fp16 [s][d] shared array.
 *  2. The causal mask cannot be applied to a coopmat accumulator (opaque
 *     lane->element mapping), so the scaled fp16 result is coopMatStore'd to a
 *     shared [s][c] scratch and then copied to global scalar-wise, applying the
 *     per-element mask. A whole-WG-tile that is entirely above the diagonal is
 *     written as -inf and skips the MMA loop (~halves prefill QK^T work).
 *
 * Dispatch: global {num_tiles_n*WG_SIZE, num_tiles_m, H_q}, local {WG_SIZE,1,1}.
 *   tileID = gl_WorkGroupID.xy (x->context, y->seq), q_h = gl_WorkGroupID.z.
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

// Bindings mirror sdpa_compute_attn_weights_tiled: attn_weights(0), q(1), k(2).
// attn_weights is written scalar-wise (masked copy), so declare scalar array.
// Coopmat is buffer-only; IO_STORAGE / K_CACHE_STORAGE are always buffer here
// (the yaml only generates the buffer/buffer variant) but are kept as params so
// the generated name carries the _buffer_buffer suffix the dispatch builds.
${layout_declare_tensor(B, "w", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_q", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}
// K-chunk trip count = head_dim / WG_TILE_K, as a spec constant (the
// Xclipse/AMD-PAL compiler crashes on a coopMatMulAdd loop with a UBO-derived
// trip count — see coopmat_mm.glsl).
${layout_declare_spec_const(C, "int", "num_k_chunks_arg", "0")}

const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint NUM_SUBGROUPS = SG_GRID_X * SG_GRID_Y;
const uint WG_SIZE = NUM_SUBGROUPS * SUBGROUP_SIZE;

const uint SG_TILE_M = WG_TILE_M / SG_GRID_Y;
const uint SG_TILE_N = WG_TILE_N / SG_GRID_X;
const uint MMAS_PER_SG_M = SG_TILE_M / MMA_M;
const uint MMAS_PER_SG_N = SG_TILE_N / MMA_N;

const uint FP16_PER_VEC4 = 4; // we read native tensors as f16vec4 (4 fp16)

// fp16 shared tiles with skew padding. A = Q [s][d], B = K^T [d][c],
// C = scaled result [s][c] scratch for the masked scalar store.
const uint A_PAD = 8;
const uint B_PAD = 8;
const uint A_ROW = WG_TILE_K + A_PAD;
const uint B_ROW = WG_TILE_N + B_PAD;

shared float16_t Ash[WG_TILE_M * A_ROW];
shared float16_t Bsh[WG_TILE_K * B_ROW];
shared float16_t Csh[WG_TILE_M * WG_TILE_N];

coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> result[MMAS_PER_SG_M][MMAS_PER_SG_N];

void main() {
    const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    const uvec2 warpInTile = uvec2(
        gl_SubgroupID % SG_GRID_X,
        gl_SubgroupID / SG_GRID_X);
    const int q_h = int(gl_WorkGroupID.z);

    // LLM layout: q_sizes WHCN {D, H_q, S, B}; k_sizes WHCN {D, H_kv, C_max, B}.
    const int D = q_sizes.x;
    const int Q_H = q_sizes.y;
    const int S = q_sizes.z;
    const int KV_H = k_sizes.y;
    const int D4 = div_up_4(D);
    const int S_aligned = align_up_4(S);
    const int context_len = input_pos + S;
    const int C4 = div_up_4(context_len);
    const int aw_row_width = C4 * 4;

    int kv_h = q_h;
    if (KV_H < Q_H) {
        kv_h = q_h / (Q_H / KV_H);
    }

    const uint M = uint(S);            // output rows (seq)
    const uint N = uint(context_len);  // output cols (context)
    const uint num_tiles_n = (N + WG_TILE_N - 1u) / WG_TILE_N;
    const uint num_tiles_m = (M + WG_TILE_M - 1u) / WG_TILE_M;
    if (tileID.x >= num_tiles_n || tileID.y >= num_tiles_m) {
        return;
    }

    const uint s_tile_base = WG_TILE_M * tileID.y;
    const uint c_tile_base = WG_TILE_N * tileID.x;

    const float16_t NEG_INF = float16_t(-1.0 / 0.0);

    // Whole-tile causal skip: if the lowest context index in this tile exceeds
    // the highest (s + input_pos), every element is masked.
    const bool tile_all_masked =
        int(c_tile_base) > (int(s_tile_base) + int(WG_TILE_M) - 1 + input_pos);
    if (tile_all_masked) {
        for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_M * WG_TILE_N;
             idx += WG_SIZE) {
            const uint ls = idx / WG_TILE_N;
            const uint lc = idx % WG_TILE_N;
            const uint gs = s_tile_base + ls;
            const uint gc = c_tile_base + lc;
            if (gs < uint(S) && gc < uint(context_len)) {
                t_attn_weights[(uint(q_h) * uint(S_aligned) + gs) *
                                   uint(aw_row_width) +
                               gc] = NEG_INF;
            }
        }
        return;
    }

    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
        }
    }

    // Per-head bases / row strides (vec4 units) for the native DHSB reads.
    const uint q_head_base = uint(q_h) * uint(D4);
    const uint q_row_stride = uint(Q_H) * uint(D4);
    const uint k_head_base = uint(kv_h) * uint(D4);
    const uint k_row_stride = uint(KV_H) * uint(D4);

    const uint VEC4_PER_CHUNK = WG_TILE_K / 4u; // d4 columns per K-chunk

    for (uint chunk = 0; chunk < uint(num_k_chunks_arg); ++chunk) {
        const uint d4_chunk = chunk * VEC4_PER_CHUNK; // d4 offset of this chunk

        // --- Stage A = Q [s][d] into fp16 shared (contiguous d) ---
        for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_M * VEC4_PER_CHUNK;
             idx += WG_SIZE) {
            const uint ls = idx / VEC4_PER_CHUNK;     // local s row
            const uint ld4 = idx % VEC4_PER_CHUNK;    // local d4 within chunk
            const uint gs = s_tile_base + ls;
            f16vec4 v = t_q[gs * q_row_stride + q_head_base + d4_chunk + ld4];
            const uint base = ls * A_ROW + ld4 * 4u;
            Ash[base + 0u] = v.x;
            Ash[base + 1u] = v.y;
            Ash[base + 2u] = v.z;
            Ash[base + 3u] = v.w;
        }

        // --- Stage B = K^T [d][c] into fp16 shared (transpose on write) ---
        for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_N * VEC4_PER_CHUNK;
             idx += WG_SIZE) {
            const uint lc = idx / VEC4_PER_CHUNK;     // local c row of K
            const uint ld4 = idx % VEC4_PER_CHUNK;    // local d4 within chunk
            const uint gc = c_tile_base + lc;
            f16vec4 v = t_k[gc * k_row_stride + k_head_base + d4_chunk + ld4];
            const uint d_base = ld4 * 4u;             // local d within chunk
            Bsh[(d_base + 0u) * B_ROW + lc] = v.x;
            Bsh[(d_base + 1u) * B_ROW + lc] = v.y;
            Bsh[(d_base + 2u) * B_ROW + lc] = v.z;
            Bsh[(d_base + 3u) * B_ROW + lc] = v.w;
        }

        barrier();

        // --- Cooperative matrix MMA: result += A * B  (B is already K^T) ---
        [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
            uint k_start = MMA_K * k;

            coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
            [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
                uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_ROW + k_start,
                    A_ROW,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
                uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
                coopMatLoad(
                    matB, Bsh,
                    k_start * B_ROW + col_b,
                    B_ROW,
                    gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
                }
            }
        }

        barrier();
    }

    // --- Scale on the fp32 accumulator, store fp16 into Csh [s][c] scratch ---
    const float16_t inv_scale_h = float16_t(inv_scale);
    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            result[i][j] = result[i][j] * inv_scale; // fp32 scalar multiply
            coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
                coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
            uint local_row = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
            uint local_col = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
            coopMatStore(
                out_tile, Csh,
                local_row * WG_TILE_N + local_col, WG_TILE_N,
                gl_CooperativeMatrixLayoutRowMajor);
        }
    }
    barrier();

    // --- Copy Csh -> global attn_weights with the per-element causal mask ---
    for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_M * WG_TILE_N;
         idx += WG_SIZE) {
        const uint ls = idx / WG_TILE_N;
        const uint lc = idx % WG_TILE_N;
        const uint gs = s_tile_base + ls;
        const uint gc = c_tile_base + lc;
        if (gs < uint(S) && gc < uint(context_len)) {
            float16_t v = Csh[idx];
            if (int(gc) > int(gs) + input_pos) {
                v = NEG_INF;
            }
            t_attn_weights[(uint(q_h) * uint(S_aligned) + gs) *
                               uint(aw_row_width) +
                           gc] = v;
        }
    }
}
