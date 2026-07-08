/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix SDPA attn*V kernel (prefill / LLM mode).
 *
 * Computes per head q_h:  out[s, d] = sum_c P[s, c] * V[c, d]
 *   P  = attn_weights (softmax output), head-contiguous [S_aligned, context_len]
 *        index (vec4 along c): (q_h * S_aligned + s) * C4 + c4
 *   V  = v_cache, DHSB [context_len, KV_H, D]
 *        index (vec4 along d): (c * KV_H + kv_h) * D4 + d4   (GQA: kv_h = q_h/(Q_H/KV_H))
 *   out= DHSB [S, Q_H, D]
 *        scalar index: (s * Q_H + q_h) * D + d
 *
 * This is the plain A*B coopmat MM (coopmat_mm.glsl) with three changes:
 *   - A (P) staging uses the head-contiguous row stride C4 + per-head base.
 *   - B (V) staging uses the DHSB head-interleaved row stride KV_H*D4 + base.
 *   - output coopMatStore uses the DHSB row stride (Q_H*D) so heads interleave;
 *     stride + head_dim are spec constants (the Xclipse/AMD-PAL compiler
 *     miscompiles coopMatStore whose stride derives from a UBO value).
 * fp16 x fp16 -> fp32 MMA. No mask / no scale (softmax already applied).
 *
 * Dispatch: global {num_tiles_n*WG_SIZE, num_tiles_m, H_q}, local {WG_SIZE,1,1}.
 *   tileID = gl_WorkGroupID.xy, q_h = gl_WorkGroupID.z.
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

// Bindings mirror sdpa_compute_out_tiled: output(0), attn_weights(1), v(2).
// Coopmat is buffer-only; IO_STORAGE / V_CACHE_STORAGE are always buffer here
// (the yaml only generates the buffer/buffer variant) but are kept as params so
// the generated name carries the _buffer_buffer suffix the dispatch builds.
${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_v", DTYPE, V_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "v_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Spec constants. inv_scale occupies id 3 (UNUSED here — softmax already
// normalized; attn*V applies no scale) so this shader stays aligned with the
// decode _coop / tiled attn*V variants that share this node's fixed spec_vars
// list and declare inv_scale at id 3. The rest are never UBO-derived (see
// coopmat_mm.glsl for the Xclipse PAL bug these work around): K-chunk trip count
// (= max_context_len/WG_TILE_K), the DHSB output row stride (Q_H*D), and
// head_dim (D) for the store column offset.
${layout_declare_spec_const(C, "float", "inv_scale_unused", "1.0")}
${layout_declare_spec_const(C, "int", "num_k_chunks_arg", "0")}
${layout_declare_spec_const(C, "int", "out_row_stride_arg", "0")}
${layout_declare_spec_const(C, "int", "head_dim_arg", "0")}

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

const uint FP16_PER_VEC4 = 8;

const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;

shared uvec4 Ash[WG_TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[WG_TILE_K * B_STRIDE_VEC4];

coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> result[MMAS_PER_SG_M][MMAS_PER_SG_N];

void main() {
    const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    const uvec2 warpInTile = uvec2(
        gl_SubgroupID % SG_GRID_X,
        gl_SubgroupID / SG_GRID_X);
    const int q_h = int(gl_WorkGroupID.z);

    // LLM layout: q_sizes WHCN {D, H_q, S, B}; v_sizes WHCN {D, H_kv, C_max, B}.
    const int D = q_sizes.x;
    const int Q_H = q_sizes.y;
    const int S = q_sizes.z;
    const int KV_H = v_sizes.y;
    const int D4 = div_up_4(D);
    const int S_aligned = align_up_4(S);
    const int context_len = input_pos + S;
    const int C4 = div_up_4(context_len);

    int kv_h = q_h;
    if (KV_H < Q_H) {
        kv_h = q_h / (Q_H / KV_H);
    }

    const uint M = uint(S);              // output rows
    const uint N = uint(head_dim_arg);   // output cols (= D)
    const uint num_tiles_n = (N + WG_TILE_N - 1u) / WG_TILE_N;
    const uint num_tiles_m = (M + WG_TILE_M - 1u) / WG_TILE_M;
    if (tileID.x >= num_tiles_n || tileID.y >= num_tiles_m) {
        return;
    }

    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
        }
    }

    const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;
    const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
    const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;

    const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;
    const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
    const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

    const uint a_row_base = WG_TILE_M * tileID.y;       // global s
    const uint b_col_base = WG_TILE_N * tileID.x;       // global d

    // Per-head bases / strides in vec4 units.
    const uint aw_head_base = uint(q_h) * uint(S_aligned) * uint(C4);
    const uint aw_row_stride = uint(C4);
    const uint v_head_base = uint(kv_h) * uint(D4);
    const uint v_row_stride = uint(KV_H) * uint(D4);

    for (uint chunk = 0; chunk < uint(num_k_chunks_arg); ++chunk) {
        const uint chunkK = chunk * WG_TILE_K;   // along context_len
        // num_k_chunks is max_context_len/WG_TILE_K (static spec const). The
        // gate guarantees context_len % WG_TILE_N == 0, hence % WG_TILE_K == 0,
        // so a chunk is either fully within context_len or fully beyond it.
        // Stage zeros for beyond-context chunks (zero contribution to the MMA).
        const bool chunk_valid = chunkK < uint(context_len);

        // --- Load A (attn_weights) tile -> shared (single pass) ---
        {
            f16vec4 v0 = f16vec4(0);
            f16vec4 v1 = f16vec4(0);
            if (chunk_valid) {
                uint row = a_row_base + a_row_offset;     // global s
                uint k_hv4 = (chunkK + a_col * FP16_PER_VEC4) / 4u;  // c, vec4
                uint base = aw_head_base + row * aw_row_stride;
                v0 = t_attn_weights[base + k_hv4];
                v1 = t_attn_weights[base + k_hv4 + 1u];
            }
            Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
        }

        // --- Load B (V) tile -> shared (single pass), row-major [c, d] ---
        {
            f16vec4 v0 = f16vec4(0);
            f16vec4 v1 = f16vec4(0);
            if (chunk_valid) {
                uint k_row = chunkK + b_row_offset;       // global c
                uint n_elem = b_col_base + b_col * FP16_PER_VEC4;  // d
                uint n4_0 = n_elem >> 2u;                 // d4
                uint base = k_row * v_row_stride + v_head_base;
                v0 = t_v[base + n4_0];
                v1 = t_v[base + n4_0 + 1u];
            }
            Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
        }

        barrier();

        // --- Cooperative matrix MMA (identical to coopmat_mm) ---
        [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
            uint k_start = MMA_K * k;

            coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
            [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
                uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
                    A_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
                uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
                coopMatLoad(
                    matB, Bsh,
                    k_start * B_STRIDE_VEC4 + col_b,
                    B_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
                }
            }
        }

        barrier();
    }

    // --- Store result: DHSB out, head-interleaved row stride (spec const) ---
    const uint out_row_stride = uint(out_row_stride_arg);  // Q_H * D
    const uint head_off = uint(q_h) * uint(head_dim_arg);  // q_h * D
    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            uint gi = WG_TILE_M * tileID.y + MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
            uint gj = WG_TILE_N * tileID.x + MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
            coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
                coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
            coopMatStore(
                out_tile, t_output,
                gi * out_row_stride + head_off + gj, out_row_stride,
                gl_CooperativeMatrixLayoutRowMajor);
        }
    }
}
