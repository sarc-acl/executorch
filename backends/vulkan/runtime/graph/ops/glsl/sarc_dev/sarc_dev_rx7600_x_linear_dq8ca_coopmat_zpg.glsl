/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, RX 7600 campaign round 2 (openspec/changes/sarc-1.5-rx7600-prefill-refine): the shipped 8da4w zpg kernel
 * with the options of sarc_dev_rx7600_dq8ca_zpg_body.glslh (A staging pad, drain tile in the A staging buffer, branch-free loop, stores
 * interleaved with the MMAs; measurement-only phase timing and ablations). The wrapper is the zpg sweep wrapper with a different body. All options default off.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable
$if PROF:
  #extension GL_ARB_shader_clock : require

#define PRECISION ${PRECISION}

$if WEIGHT_NBITS == 4:
  #define WEIGHT_INT4

// INTERVENTION G: when the A staging thread map exactly covers the workgroup
// (A_ACTIVE_THREADS == WG_SIZE) the `a_active` guard is statically always true,
// but the driver compiler does not fold it -- gl_LocalInvocationID.x's bound
// comes from a spec constant, so the comparison survives into the hot loop as a
// real branch. Set A_MAP_FULL only for tiles where the equality has been
// checked arithmetically; the yaml records the arithmetic per variant.
$if A_MAP_FULL:
  #define A_ALWAYS_ACTIVE

$if HAS_BIAS:
  #define HAS_BIAS

$if PROF:
  #define RX_PROF

$if CSH_IN_ASH:
  #define RX_CSH_IN_ASH

$if BF:
  #define RX_BF

$if UV4:
  #define RX_UV4

$if ST_A >= 0:
  #define RX_ST_A ${ST_A}

$if ST_B >= 0:
  #define RX_ST_B ${ST_B}

$if ABL > 0:
  #define RX_ABL ${ABL}

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

layout(std430) buffer;

#include "common.glslh"

// Bindings — match add_linear_dqa_qw_node arg order:
//   output(0), fp_input(1), packed_int8_input(2), int_input_sums(3 - unused),
//   input_scales(4), input_zps(5), packed_weight(6), weight_sums(7),
//   weight_scales(8), bias(9).
${layout_declare_tensor(B, "w", "t_output",              "half", IO_STORAGE, is_scalar_array=True)}
// t_input is unread here -- the activations arrive already quantized in
// t_packed_int8_input -- but stays declared so the binding layout matches the
// dispatch site. It tracks IO_STORAGE so the two IO tensors stay consistent.
${layout_declare_tensor(B, "r", "t_input",               "half", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int",  "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_weight",       "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums",         "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
// INT4 only; inert (0) for INT8 so the dispatcher's spec list lines up.
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// Tile geometry
const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
$if A_MULTI_BLOCK:
  #define A_MULTI_BLOCK
const uint A_BLOCKS = ${A_BLOCKS};
const uint A_PAD_U32 = ${A_PAD};

#include "sarc_dev_rx7600_dq8ca_zpg_body.glslh"
