/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define NUM_WORKERS_PER_WG 64

$if MODE == "llm":
  #define HAS_INPUT_POS

#define IN_DTYPE ${IN_DTYPE}
#define OUT_DTYPE ${OUT_DTYPE}
#define SOFTMAX_IN_VEC4_T ${texel_load_type(IN_DTYPE, STORAGE)}
#define SOFTMAX_ACC_T ${texel_load_component_type(IN_DTYPE, STORAGE)}
#define VEC4_T ${texel_load_type(OUT_DTYPE, STORAGE)}
#define T ${texel_load_component_type(OUT_DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(STORAGE, [IN_DTYPE, OUT_DTYPE])}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights_softmax", OUT_DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", IN_DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
$if MODE == "llm":
  ${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Shared memory for cooperative max finding and exp sum reduction.
// For fused SDPA, reductions happen in fp32 for numerical stability.
shared SOFTMAX_ACC_T shared_max[NUM_WORKERS_PER_WG];
shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];

SOFTMAX_IN_VEC4_T load_attn_weights_c4(
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  return t_attn_weights[(q_h * S * C4) + (s * C4) + c4];
#else
  return SOFTMAX_IN_VEC4_T(texelFetch(t_attn_weights, ivec3(c4, s, q_h), 0));
#endif
}

void store_attn_weights_softmax_c4(
    const VEC4_T out_texel,
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  t_attn_weights_softmax[(q_h * S * C4) + (s * C4) + c4] = out_texel;
#else
  imageStore(t_attn_weights_softmax, ivec3(c4, s, q_h), out_texel);
#endif
}

/*
 * 3-pass numerically stable softmax over the context_len dimension of
 * attention weights.
 *
 * LLM SDPA (HAS_INPUT_POS):
 *   reads VEC4_T (input dtype), reduces in T, writes VEC4_T.
 *   attn_weights S dim is padded to S_aligned.
 *   current context_len = input_pos + S.
 *
 * Fused SDPA (!HAS_INPUT_POS):
 *   reads vec4 (fp32 from QK), reduces in fp32, writes VEC4_T (input dtype).
 *   attn_weights S dim is not padded.
 *   context_len = k_sizes.y.
 *
 * Dispatch: (1, S, H * B) — for LLM (batch=1), H * B == H_q.
 */
void main() {
  const int worker_id = int(gl_LocalInvocationID.x);

  // Index along attention weight's sequence_len dim
  const int s = int(gl_GlobalInvocationID.y);
  // For LLM: q_head index. For fused: combined batch*H + head index.
  const int q_h = int(gl_GlobalInvocationID.z);

#ifdef HAS_INPUT_POS
  // LLM: q_sizes is WHCN {D, H_q, S, B}
  const int Q_H = q_sizes.y;
  const int S = q_sizes.z;
#else
  // Fused: q_sizes is WHCN {D, S, H, B}
  const int Q_H = q_sizes.z;
  const int S = q_sizes.y;
#endif
  const int S_aligned = align_up_4(S);

#ifdef HAS_INPUT_POS
  // manually determine size of the context_len dim of the attention weight.
  // The "actual" tensor sizes may have been aligned to a multiple of 4 to allow
  // memory loads to be aligned to texel boundaries.
  const int context_len = input_pos + S;
#else
  const int context_len = k_sizes.y;
#endif
  const int context_texel_len = div_up_4(context_len);

  // LLM: attn_weights S dim is padded to S_aligned; fused: not padded.
#ifdef HAS_INPUT_POS
  const int attn_S = S_aligned;
#else
  const int attn_S = S;
#endif

  // bounds check — q_h bound is Q_H * batch_size; for LLM (batch=1) this
  // equals Q_H, for fused this equals H * B.
  if (s >= S || q_h >= Q_H * q_sizes.w) {
    return;
  }

  const int context_len_aligned_down = context_len - mod_4(context_len);
  const int C4_limit = div_4(context_len_aligned_down);

  // Causal-mask truncation. In LLM mode the QK^T shader has already written
  // -inf to every element with c > s + input_pos, so for row s only the first
  // (s + input_pos + 1) columns carry information: a masked element
  // contributes exp(-inf - max) == 0 to the sum, cannot be the row max unless
  // the whole row is masked, and normalizes to exactly 0. All three passes can
  // therefore stop at reduce_len instead of context_len, which on a full
  // prefill halves the bytes read (the rows are a triangle, not a rectangle).
  //
  // The STORE still has to cover the whole context_len row: attn*V stages
  // every chunk with chunkK < context_len and multiplies it by V, so the
  // masked tail must be written as zero rather than left stale. Writing zero
  // is cheaper than reading + exp + writing, and needs no load.
  //
  // Fused mode has no input_pos and gets its mask from an attn_mask bias
  // instead, so no truncation is valid there.
#ifdef HAS_INPUT_POS
  const int reduce_len = min(context_len, s + input_pos + 1);
#else
  const int reduce_len = context_len;
#endif
  const int reduce_texel_len = div_up_4(reduce_len);
  const int reduce_len_aligned_down = reduce_len - mod_4(reduce_len);
  const int R4_limit = div_4(reduce_len_aligned_down);

  // =========================================================================
  // Pass 1: Find the maximum value across the row for numerical stability.
  // Without this, exp(x) can overflow float32 when x > ~88.7.
  // =========================================================================

  SOFTMAX_ACC_T local_max = SOFTMAX_ACC_T(-1.0 / 0.0); // -infinity

  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {
    SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, attn_S, Q_H);

    for (int comp = 0; comp < 4; comp++) {
      local_max = max(local_max, SOFTMAX_ACC_T(in_texel[comp]));
    }
  }
  if (worker_id == 0) {
    for (int c4 = R4_limit; c4 < reduce_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, attn_S, Q_H);

      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < reduce_len) {
          local_max = max(local_max, SOFTMAX_ACC_T(in_texel[comp]));
        }
      }
    }
  }

  shared_max[worker_id] = local_max;

  memoryBarrierShared();
  barrier();

  // Tree reduction to find the global max
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_max[worker_id] = max(
          shared_max[worker_id], shared_max[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }

  const SOFTMAX_ACC_T global_max = shared_max[0];

  // =========================================================================
  // Pass 2: Compute sum(exp(x - max)) using the global max for stability
  // =========================================================================

  SOFTMAX_ACC_T local_exp_sum = SOFTMAX_ACC_T(0);

  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {
    SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, attn_S, Q_H);

    for (int comp = 0; comp < 4; comp++) {
      local_exp_sum += exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
    }
  }
  if (worker_id == 0) {
    for (int c4 = R4_limit; c4 < reduce_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, attn_S, Q_H);

      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < reduce_len) {
          local_exp_sum += exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
        }
      }
    }
  }

  shared_exp_sum[worker_id] = local_exp_sum;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall exp sum
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_exp_sum[worker_id] = shared_exp_sum[worker_id] +
          shared_exp_sum[worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }

  local_exp_sum = shared_exp_sum[0];

  // =========================================================================
  // Pass 3: Normalize each element: out = exp(x - max) / sum(exp(x - max))
  // =========================================================================

  // Fully-inside-the-prefix texels: load, normalize, store.
  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {
    SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, attn_S, Q_H);

    VEC4_T out_texel;
    [[unroll]] for (int comp = 0; comp < 4; comp++) {
      out_texel[comp] = T(
          exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
    }
    store_attn_weights_softmax_c4(
        out_texel, c4, s, q_h, context_texel_len, attn_S, Q_H);
  }

  // The single texel that straddles reduce_len, if reduce_len is not a
  // multiple of 4. Its masked lanes normalize to 0, same as the tail below.
  if (worker_id == 0) {
    for (int c4 = R4_limit; c4 < reduce_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, attn_S, Q_H);

      VEC4_T out_texel = VEC4_T(0);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < reduce_len) {
          out_texel[comp] = T(
              exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
        }
      }
      store_attn_weights_softmax_c4(
          out_texel, c4, s, q_h, context_texel_len, attn_S, Q_H);
    }
  }

  // Causally-masked tail: every element normalizes to exactly 0, so write zero
  // WITHOUT loading the input. This is the read the truncation saves in pass 3,
  // and the store attn*V depends on (it stages the whole context_len row).
  // Spread across all workers, unlike the straddling texel above.
  for (int c4 = reduce_texel_len + worker_id; c4 < context_texel_len;
       c4 += NUM_WORKERS_PER_WG) {
    store_attn_weights_softmax_c4(
        VEC4_T(0), c4, s, q_h, context_texel_len, attn_S, Q_H);
  }
}
