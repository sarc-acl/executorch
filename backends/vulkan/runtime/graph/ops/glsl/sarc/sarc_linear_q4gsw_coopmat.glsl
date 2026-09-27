/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC fp16 q4gsw (4-bit group-symmetric weight) cooperative-matrix linear.
 *
 * This file is only the template header: bindings, spec constants and the
 * per-variant tile geometry. The kernel body is the untemplated
 * sarc_linear_q4gsw_coopmat_body.glslh, shared with the development sweep
 * wrapper (glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.glsl), which must stay
 * byte-identical to this file apart from this comment block.
 *
 * Loop structure: dbuf4 ("store-first", prefetch-peeled): prologue prefetches
 * chunk 0; per iteration barrier -> prefetch(next) -> MMA(cur) -> store(next).
 *
 * Feature defines (yaml parameters, all default off):
 *   ACC_FP32        fp32 accumulator (RDNA3 780M: maps 1:1 onto
 *                   v_wmma_f32_16x16x16_f16; Orin large-K accuracy)
 *   ACC_GROUP_FP32  fp16 accumulate per quantization group, fp32 total
 *                   (GeForce: full-rate fp16 MMA without the long-K error)
 *   CSH_IN_ASH      texture3d drain staged in dead Ash (RDNA3 occupancy)
 *   FRAG_LAYOUT     fragment-contiguous LDS, no padding (Intel Xe2)
 *   IMG_A / IMG_W   storage-image loads for A / weights (Intel Xe2)
 *   MMA_M = 8       Intel Xe2 exposes fp16 coopmat only at 8x16x16
 *
 * Hard preconditions (no shape checks in the shader), enforced by
 * sarc::select() at dispatch: M % WG_TILE_M == 0, N % WG_TILE_N == 0,
 * K % WG_TILE_K == 0, group_size % WG_TILE_K == 0, no bias, batch 1.
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

// ACC_GROUP_FP32 (RTX 4070 Ti SUPER, 2026-09-26): accumulate each quantization
// group in fp16 at the full fp16 MMA rate, then add it into an fp32 total and
// restart, so an fp16 run never exceeds one group. The GeForce fp16-accumulate
// MMA loses accuracy over long K (8B w2, K = 14336: max |err| 1.49, over the
// 0.5 tolerance; already 0.45 at K = 4096); with this, 0.08 / 0.04. Plain fp32
// accumulation (0.04) runs at half the tensor rate: 0.63x vs 0.92x speed.
$if ACC_GROUP_FP32:
  #define ACC_GROUP_FP32

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output",         "half", IO_STORAGE, is_scalar_array=True)}
$if IMG_A and IO_STORAGE == "texture3d":
  ${layout_declare_image(B, "r", "t_input", "half")}
$else:
  ${layout_declare_tensor(B, "r", "t_input",          "half", IO_STORAGE, is_scalar_array=False)}
$if IMG_W and WEIGHT_STORAGE == "texture2d":
  ${layout_declare_image(B, "r", "t_packed_weight", "int", image_ndim=2)}
$else:
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
#if defined(ACC_FP32) && defined(ACC_GROUP_FP32)
#error "ACC_FP32 (fp32 accumulate) and ACC_GROUP_FP32 (fp16 per group, fp32 total) are exclusive"
#endif
#ifdef ACC_FP32
#define ACC_T float
#else
#define ACC_T float16_t
#endif

$if CSH_IN_ASH:
  #define CSH_IN_ASH

$if SH_F16V4:
  #define SH_F16V4

$if FRAG_LAYOUT:
  #define FRAG_LAYOUT

$if IMG_A and IO_STORAGE == "texture3d":
  #define IMG_A

$if IMG_W and WEIGHT_STORAGE == "texture2d":
  #define IMG_W

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
