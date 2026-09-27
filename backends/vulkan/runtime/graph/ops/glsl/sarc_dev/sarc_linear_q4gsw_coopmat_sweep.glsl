/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone: sweep twin of glsl/sarc/sarc_linear_q4gsw_coopmat.glsl.
 * Everything from #version on must stay byte-identical to that file (checked
 * by sarc/tools/check.sh); only the template name (this file name) and the
 * variant list in the yaml differ.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output",         "half", IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",          "half", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales",  "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",           "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

$if ACC_FP32:
  #define ACC_FP32

// Accumulator element type. ACC_FP32 (RDNA3, 780M roofline study 2026-09-26):
// v_wmma_f16_16x16x16_f16 takes/returns one fp16 per VGPR, so a packed
// coopmat<float16_t> accumulator is repacked with v_mov_b16 around every WMMA
// (~350 moves per 32 WMMAs in the t128x128k32g42s32 loop); the fp32
// accumulator maps 1:1 onto v_wmma_f32_16x16x16_f16. Roofline: matrix
// fp16->fp32 14.77 vs fp16->fp16 10.96 TFLOP/s on the 780M.
#ifdef ACC_FP32
#define ACC_T float
#else
#define ACC_T float16_t
#endif

$if CSH_IN_ASH:
  #define CSH_IN_ASH

// --- Tile geometry (from yaml; per-variant tile-sweep candidate) ---
const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};

#include "sarc_linear_q4gsw_coopmat_body.glslh"
