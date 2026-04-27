/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Phase 4 prototype: KHR Cooperative Matrix int8 linear shader.
 *
 * Computes: C = A @ B   (A: int8 [M,K] row-major, B: int8 [K,N] row-major,
 *                        C: int32 [M,N] row-major)
 *
 * 16x16x16 int8xint8 -> int32 cooperative matrix MMA. Macro tile 64x64x32.
 *
 * Intended scope: kernel-throughput microbenchmark only. No scales,
 * zero-points, dequantization, or biases — those layers belong to a follow-up
 * design once int8 coopmat throughput is proven on this device.
 *
 * Layout choices made for the prototype (NOT the production
 * pack_q8_linear_weight format):
 *   - A is int8 [M, K] row-major in a buffer of int (4 int8 per int element
 *     along K), so element (m, k) is `A_buf[m*K/4 + k/4][k%4]`.
 *   - B is int8 [K, N] row-major in a buffer of int (4 int8 per int element
 *     along N), so element (k, n) is `B_buf[k*N/4 + n/4][n%4]`.
 *   - C is int32 [M, N] row-major, one int per element.
 *
 * Production integration with the existing pack_q8_linear_weight layout is a
 * Phase 5 concern. See yanwen_docs/agent_reports/int8_coopmat_exploration_rdna3.md
 * for the layout migration plan.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

layout(std430) buffer;

// Bindings: output(0), A(1), B(2)
${layout_declare_tensor(B, "w", "t_output", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_mat1", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight", "int", "buffer", is_scalar_array=True)}

// UBOs
${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Cooperative matrix tile (hardware tile)
const uint lM = 16;
const uint lN = 16;
const uint lK = 16;

// Macro tile per workgroup (mirrors linear_coopmat.glsl 64x64x32 default)
const uint TILE_M = 64;
const uint TILE_N = 64;
const uint TILE_K = 32;

const uint WG_WIDTH = 2;   // subgroups along N
const uint WG_HEIGHT = 2;  // subgroups along M
const uint NUM_SUBGROUPS = WG_WIDTH * WG_HEIGHT;
const uint INVOCATIONS = 64 * NUM_SUBGROUPS;  // subgroup size 64

const uint C_ROWS = TILE_M / WG_HEIGHT / lM;  // 2
const uint C_COLS = TILE_N / WG_WIDTH / lN;   // 2

// Shared memory: hold the A and B macro tiles in int8 row-major.
// 16 int8 per uvec4 (128 bits). The +1 uvec4 row padding mirrors the fp16
// coopmat shader and breaks the 32-byte stride that otherwise creates LDS
// bank conflicts on RDNA3 (32 banks * 32 bits).
const uint INT8_PER_VEC4 = 16;

const uint A_STRIDE_VEC4 = TILE_K / INT8_PER_VEC4 + 1u;  // = 3 for TILE_K=32
const uint B_STRIDE_VEC4 = TILE_N / INT8_PER_VEC4 + 1u;  // = 5 for TILE_N=64

shared uvec4 Ash[TILE_M * A_STRIDE_VEC4];  // 64 * 3 = 192 uvec4 = 3072 B
shared uvec4 Bsh[TILE_K * B_STRIDE_VEC4];  // 32 * 5 = 160 uvec4 = 2560 B

coopmat<int32_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];

void main() {
    const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    const uvec2 warpInTile = uvec2(
        gl_SubgroupID % WG_WIDTH,
        gl_SubgroupID / WG_WIDTH);

    // mat1 is declared as `int [M, K/4]` (each int packs 4 int8 along K),
    // so the size UBO's x dimension is the int32 count, not the int8 count.
    // Convert to logical int8 K for the chunk loop.
    const uint K_int = uint(mat1_sizes.x);
    const uint M = uint(mat1_sizes.y);
    const uint K = K_int * 4u;
    const uint N = uint(out_sizes.x);
    const uint N_int = N / 4u;

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            result[i][j] = coopmat<int32_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0);
        }
    }

    const uint INVS_PER_ROW_A = TILE_K / INT8_PER_VEC4;  // 2
    const uint INVS_PER_ROW_B = TILE_N / INT8_PER_VEC4;  // 4

    const uint a_row_base = TILE_M * tileID.y;
    const uint b_col_base = TILE_N * tileID.x;

    for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {

        // Stage A macro tile [TILE_M, TILE_K] from t_mat1 into Ash.
        // t_mat1[row * K_int + k_int] holds 4 int8 along K.
        for (uint idx = gl_LocalInvocationID.x;
             idx < TILE_M * INVS_PER_ROW_A;
             idx += INVOCATIONS) {
            uint a_col = idx % INVS_PER_ROW_A;
            uint a_row_offset = idx / INVS_PER_ROW_A;
            uint row = a_row_base + a_row_offset;
            uint k_int_base = (chunkK + a_col * INT8_PER_VEC4) / 4u;

            // Each uvec4 holds 16 int8 = 4 ints.
            Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
                t_mat1[row * K_int + k_int_base + 0],
                t_mat1[row * K_int + k_int_base + 1],
                t_mat1[row * K_int + k_int_base + 2],
                t_mat1[row * K_int + k_int_base + 3]);
        }

        // Stage B macro tile [TILE_K, TILE_N] from t_weight into Bsh.
        // t_weight[k * N_int + n_int] holds 4 int8 along N.
        for (uint idx = gl_LocalInvocationID.x;
             idx < TILE_K * INVS_PER_ROW_B;
             idx += INVOCATIONS) {
            uint b_col = idx % INVS_PER_ROW_B;
            uint b_row_offset = idx / INVS_PER_ROW_B;
            uint k_row = chunkK + b_row_offset;
            uint n_int_base = (b_col_base + b_col * INT8_PER_VEC4) / 4u;

            Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
                t_weight[k_row * N_int + n_int_base + 0],
                t_weight[k_row * N_int + n_int_base + 1],
                t_weight[k_row * N_int + n_int_base + 2],
                t_weight[k_row * N_int + n_int_base + 3]);
        }

        barrier();

        // --- Cooperative matrix MMA over the macro K-step ---
        [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k) {
            uint k_start = lK * k;

            coopmat<int8_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                uint row_a = lM * (C_ROWS * warpInTile.y + i);
                // Offset is in uvec4 units; row stride accounts for the padded row.
                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_STRIDE_VEC4 + k_start / INT8_PER_VEC4,
                    A_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            coopmat<int8_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
                uint col_b = lN * (C_COLS * warpInTile.x + j);
                coopMatLoad(
                    matB, Bsh,
                    k_start * B_STRIDE_VEC4 + col_b / INT8_PER_VEC4,
                    B_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
                }
            }
        }

        barrier();
    }

    // --- Store int32 result ---
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * (C_ROWS * warpInTile.y + i);
            uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);
            coopMatStore(
                result[i][j], t_output,
                gi * N + gj, N,
                gl_CooperativeMatrixLayoutRowMajor);
        }
    }
}
