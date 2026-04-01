/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix int8 GEMM shader.
 * Computes: D = A * B (pure integer matmul, no dequantization)
 *
 * A: [M, K] row-major buffer (int8)
 * B: [K, N] row-major buffer (int8)
 * D: [M, N] row-major buffer (int32 accumulator output)
 *
 * Uses shared memory tiling with double-buffered prefetch,
 * same architecture as the FP16 addmm_khr_cm.glsl shader.
 */

#version 450 core
#pragma use_vulkan_memory_model

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : enable

#define A_BITS 8
#define A_SCALAR_TYPE uint8_t
#define C_SCALAR_TYPE uint32_t

layout(std430) buffer;

// Buffer bindings: D (float output — int32 accumulator cast to float), A (uvec4 input), B (uvec4 input)
layout(set = 0, binding = 0) buffer restrict writeonly DBuffer {
    float t_D[];
};
layout(set = 0, binding = 1) buffer restrict readonly AV4Buffer {
    uvec4 t_A_v4[];
};
layout(set = 0, binding = 2) buffer restrict readonly BV4Buffer {
    uvec4 t_B_v4[];
};

// Push constants: K and strides only (no alpha/beta for pure integer GEMM)
layout(push_constant) uniform restrict Block {
    uint K;
    uint strideA;
    uint strideB;
    uint strideD;
};

// Workgroup size set via specialization constants (IDs 0-2)
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Cooperative matrix tile dimensions (IDs 3+)
layout(constant_id = 3) const uint lM = 16;
layout(constant_id = 4) const uint lN = 16;
layout(constant_id = 5) const uint lK = 16;
layout(constant_id = 6) const uint TILE_M = 128;
layout(constant_id = 7) const uint TILE_N = 128;
layout(constant_id = 8) const uint TILE_K = 32;
layout(constant_id = 9) const uint A_ROW_LEN = 32;
layout(constant_id = 10) const uint A_NUM_ROWS = 128;
layout(constant_id = 11) const uint B_ROW_LEN = 128;
layout(constant_id = 12) const uint B_NUM_ROWS = 32;
layout(constant_id = 13) const uint BColMajor_val = 0;

const bool BColMajor = (BColMajor_val != 0);

// Elements per uvec4 load: 16 bytes / 1 byte per int8 = 16 elements
const uint ELEMENTS_PER_VEC4 = 16 / (A_BITS / 8);
const uint ROW_PAD_SH = ELEMENTS_PER_VEC4;

// Shared memory with skew padding to avoid bank conflicts
shared uvec4 Ash[A_NUM_ROWS * (A_ROW_LEN + ROW_PAD_SH) / ELEMENTS_PER_VEC4];
shared uvec4 Bsh[B_NUM_ROWS * (B_ROW_LEN + ROW_PAD_SH) / ELEMENTS_PER_VEC4];

const uint WORKGROUP_WIDTH_IN_SUBGROUPS = 4;
const uint WORKGROUP_HEIGHT_IN_SUBGROUPS = 2;
const uint NUM_SUBGROUPS = WORKGROUP_WIDTH_IN_SUBGROUPS * WORKGROUP_HEIGHT_IN_SUBGROUPS;
const uint INVOCATIONS_PER_WORKGROUP = 32 * NUM_SUBGROUPS;

const uint C_ROWS = TILE_M / WORKGROUP_HEIGHT_IN_SUBGROUPS / lM;
const uint C_COLS = TILE_N / WORKGROUP_WIDTH_IN_SUBGROUPS / lN;

// Int32 accumulator cooperative matrices
coopmat<C_SCALAR_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];

uint coordToOffset(uint i, uint j, uint stride, bool colMajor) {
    return colMajor ? (stride * j + i) : (stride * i + j);
}

void main() {
    uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    uvec2 warpInTile = uvec2(
        gl_SubgroupID % WORKGROUP_WIDTH_IN_SUBGROUPS,
        gl_SubgroupID / WORKGROUP_WIDTH_IN_SUBGROUPS);

    // Initialize result to zero (integer)
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            result[i][j] = coopmat<C_SCALAR_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0);
        }
    }

    uint chunkK = 0;

    // Prefetch A tile from global to registers
    const uint INVS_PER_ROW_A = A_ROW_LEN / ELEMENTS_PER_VEC4;
    uint atilek = ELEMENTS_PER_VEC4 * (gl_LocalInvocationID.x % INVS_PER_ROW_A);

    uvec4 temp_A[A_NUM_ROWS / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
    uint gabase = coordToOffset(TILE_M * tileID.y, chunkK, strideA, false);
    [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
        uint atilei = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
        temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)] =
            t_A_v4[(gabase + strideA * atilei + atilek) / ELEMENTS_PER_VEC4];
    }

    // Prefetch B tile from global to registers
    const uint INVS_PER_ROW_B = B_ROW_LEN / ELEMENTS_PER_VEC4;
    uint btilej = ELEMENTS_PER_VEC4 * (gl_LocalInvocationID.x % INVS_PER_ROW_B);

    uvec4 temp_B[B_NUM_ROWS / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];
    uint gbbase = coordToOffset(chunkK, TILE_N * tileID.x, strideB, BColMajor);
    [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
        uint btilek = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
        temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)] =
            t_B_v4[(gbbase + strideB * btilek + btilej) / ELEMENTS_PER_VEC4];
    }

    // Main K-loop: tile over K dimension
    for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {
        bool last = ((chunkK + TILE_K) >= K);

        const uint STRIDE_A_SH = (A_ROW_LEN + ROW_PAD_SH);

        barrier();

        // Store A from registers to shared memory
        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint si = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            Ash[(STRIDE_A_SH * si + atilek) / ELEMENTS_PER_VEC4] =
                temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
        }

        const uint STRIDE_B_SH = (B_ROW_LEN + ROW_PAD_SH);

        // Store B from registers to shared memory
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint sk = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            Bsh[(STRIDE_B_SH * sk + btilej) / ELEMENTS_PER_VEC4] =
                temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];
        }

        barrier();

        // Prefetch next A tile
        uint gabase_next = coordToOffset(TILE_M * tileID.y, chunkK + TILE_K, strideA, false);
        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint atilei = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            if (!last) {
                temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)] =
                    t_A_v4[(gabase_next + strideA * atilei + atilek) / ELEMENTS_PER_VEC4];
            }
        }

        // Prefetch next B tile
        uint gbbase_next = coordToOffset(chunkK + TILE_K, TILE_N * tileID.x, strideB, BColMajor);
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint btilek = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            if (!last) {
                temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)] =
                    t_B_v4[(gbbase_next + strideB * btilek + btilej) / ELEMENTS_PER_VEC4];
            }
        }

        // Cooperative matrix multiply-accumulate from shared memory
        [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k) {
            uint sk = lK * k;

            // Load int8 A tiles
            coopmat<A_SCALAR_TYPE, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                uint si = lM * (C_ROWS * warpInTile.y + i);
                coopMatLoad(
                    matA[i], Ash,
                    coordToOffset(si, sk, STRIDE_A_SH, false) / ELEMENTS_PER_VEC4,
                    STRIDE_A_SH / ELEMENTS_PER_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            // Load int8 B tile and multiply-accumulate into int32
            coopmat<A_SCALAR_TYPE, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
                uint sj = lN * (C_COLS * warpInTile.x + j);
                coopMatLoad(
                    matB, Bsh,
                    coordToOffset(sk, sj, STRIDE_B_SH, BColMajor) / ELEMENTS_PER_VEC4,
                    STRIDE_B_SH / ELEMENTS_PER_VEC4,
                    BColMajor ? gl_CooperativeMatrixLayoutColumnMajor : gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
                }
            }
        }
    }

    // Convert uint32 accumulator to float and store.
    // coopMatStore of float to global memory works reliably (proven by FP16 shader).
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * (C_ROWS * warpInTile.y + i);
            uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);

            // Convert uint32 cooperative matrix to float
            coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> float_result =
                coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(result[i][j]);

            coopMatStore(
                float_result, t_D,
                gi * strideD + gj,
                strideD,
                gl_CooperativeMatrixLayoutRowMajor);
        }
    }
}
