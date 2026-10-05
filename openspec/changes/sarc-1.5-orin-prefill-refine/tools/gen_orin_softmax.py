#!/usr/bin/env python3
"""gen_orin_softmax.py: the SARC LLM softmax that reads its row once, for the Jetson Orin (dev-zone twin of
glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_4070ti.glsl, which is read, never written).

  glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_orin.{glsl,yaml}
      sarc_sdpa_attn_weights_softmax_buffer_half_orin_{l,s}{2,8,16,32}[e]

Why. The softmax makes three passes over a row (maximum, sum of exp, normalise) and loads the row from the
buffer in each: on a 2048-token prefill of the 1B model that is 3 x 134 MB read and 134 MB written per layer,
6.5 + 2.3 ms at the fresh DRAM roofs (62.1 GB/s read, 58.0 write); measured 8.75 ms (4070ti_nzf). The time is
the traffic. A worker owns every 64th texel of its row, 8 texels for a 2048-token row, so it can keep them:

  l<N>  the first N texels a worker loads in pass 1 stay in a local array; passes 2 and 3 take them from there
  s<N>  the same in shared memory (each worker reads only the slots it wrote: no barrier is added)
  e     pass 2 also keeps exp(x - max), so pass 3 only divides

A row longer than 64 x 4 x N elements is handled as before beyond the kept texels (they are loaded again), so
no context length is excluded. No test case has a row longer than 2048 elements, so with N >= 8 that path would
never run in a test: the variants with N = 2 (l2, l2e, s2, s2e) exist to exercise it (every row longer than 512
elements takes it) in the bit comparison; they are not candidates. Every variant is NZ + ACC32 (the arithmetic of 4070ti_nzf: fp32 reduction, zero
tail limited to what the SARC attn*V kernels read). The values, the operations and their order are those of
4070ti_nzf, so the output is meant to be bit-identical to it; the gate checks that, it is not assumed.
Selected by ET_VK_SARC_SOFTMAX_VARIANT=orin_<variant> through the release hook Override::softmax_variant."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from devzone import write_new
ET = pathlib.Path(__file__).resolve().parents[4]
g = ET / "backends/vulkan/runtime/graph/ops/glsl"
def sub(text, old, new, count=1):
    n = text.count(old); assert n == count, f"anchor occurs {n} times, expected {count}: {old[:70]!r}"
    return text.replace(old, new)
s = (g / "sarc_dev/sarc_sdpa_attn_weights_softmax_4070ti.glsl").read_text()
s = s[s.index("#version 450 core"):]
s = sub(s, "$if NZ:\n  #define NZ_TAIL\n",
        "$if NZ:\n  #define NZ_TAIL\n\n"
        "// orin: texels of its row a worker keeps after pass 1 (local array, or shared memory).\n"
        "#define CACHE_N ${CACHE_N}\n"
        "$if CACHE == \"shared\":\n  #define CACHE_SHARED\n"
        "$if CACHE_EXP:\n  #define CACHE_EXP\n")
s = sub(s, "shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];\n",
        "shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];\n"
        "#ifdef CACHE_SHARED\n"
        "// orin: a worker reads only the slots it wrote itself, so no barrier guards them.\n"
        "shared vec4 row_cache[NUM_WORKERS_PER_WG * CACHE_N];\n"
        "#define CACHE_AT(j) row_cache[worker_id * CACHE_N + (j)]\n"
        "#else\n"
        "#define CACHE_AT(j) row_cache[j]\n"
        "#endif\n")
s = sub(s, "  SOFTMAX_ACC_T local_max = SOFTMAX_ACC_T(-1.0 / 0.0); // -infinity\n",
        "  SOFTMAX_ACC_T local_max = SOFTMAX_ACC_T(-1.0 / 0.0); // -infinity\n"
        "#ifndef CACHE_SHARED\n  vec4 row_cache[CACHE_N];\n#endif\n")
LOOP = "  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {\n"
LOAD = ("    SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(\n"
        "        c4, s, q_h, context_texel_len, attn_S, Q_H);\n")
LOOPJ = "  for (int c4 = worker_id, j = 0; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG, ++j) {\n"
KEPT = ("    SOFTMAX_IN_VEC4_T in_texel;\n"
        "    if (j < CACHE_N) {\n"
        "      in_texel = SOFTMAX_IN_VEC4_T(CACHE_AT(j));\n"
        "    } else {\n"
        "      in_texel = load_attn_weights_c4(\n"
        "          c4, s, q_h, context_texel_len, attn_S, Q_H);\n"
        "    }\n")
p = s.split(LOOP + LOAD); assert len(p) == 4, len(p)
# pass 1: load and keep
p[1] = LOOPJ + LOAD + "    if (j < CACHE_N) {\n      CACHE_AT(j) = vec4(in_texel);\n    }\n" + p[1]
# pass 2: from the kept texels; with CACHE_EXP the exponentials replace them
p[2] = LOOPJ + KEPT + sub(p[2], """
    for (int comp = 0; comp < 4; comp++) {
      local_exp_sum += exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
    }
""", """
#ifdef CACHE_EXP
    vec4 e4;
    for (int comp = 0; comp < 4; comp++) {
      const SOFTMAX_ACC_T e = exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
      local_exp_sum += e;
      e4[comp] = float(e);
    }
    if (j < CACHE_N) {
      CACHE_AT(j) = e4;
    }
#else
    for (int comp = 0; comp < 4; comp++) {
      local_exp_sum += exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
    }
#endif
""")
# pass 3
p[3] = LOOPJ + sub(p[3], """
    VEC4_T out_texel;
    [[unroll]] for (int comp = 0; comp < 4; comp++) {
      out_texel[comp] = T(
          exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
    }
""", """
    VEC4_T out_texel;
#ifdef CACHE_EXP
    if (j < CACHE_N) {
      const vec4 e4 = CACHE_AT(j);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        out_texel[comp] = T(SOFTMAX_ACC_T(e4[comp]) / local_exp_sum);
      }
    } else {
      const SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, attn_S, Q_H);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        out_texel[comp] = T(
            exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
      }
    }
#else
""" + KEPT + """    [[unroll]] for (int comp = 0; comp < 4; comp++) {
      out_texel[comp] = T(
          exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
    }
#endif
""")
s = p[0] + p[1] + p[2] + p[3]
HDR = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Jetson Orin (generated by gen_orin_softmax.py;
 * openspec/changes/sarc-1.5-orin-prefill-refine): sarc_sdpa_attn_weights_softmax_4070ti.glsl in which a
 * worker keeps the first CACHE_N texels of its row after pass 1 (local array or shared memory) instead of
 * loading them again in passes 2 and 3, and optionally (CACHE_EXP) keeps exp(x - max) after pass 2.
 * Same values, operations and order as the 4070ti variant with the same NZ / ACC32.
 */

"""
F = "sarc_sdpa_attn_weights_softmax_orin"
V = [(c, n, e) for c in ("local", "shared") for n in (2, 8, 16, 32) for e in (False, True)]
write_new(g / f"sarc_dev/{F}.glsl", HDR + s)
write_new(g / f"sarc_dev/{F}.yaml", f"""# SARC development zone, Jetson Orin: LLM softmax variants that read a row once (generated by gen_orin_softmax.py). Not shipped.

{F}:
  parameter_names_with_default_values:
    IN_DTYPE: half
    OUT_DTYPE: half
    STORAGE: buffer
    MODE: llm
    NZ: true
    ACC32: true
    CACHE: local
    CACHE_N: 8
    CACHE_EXP: false
  shader_variants:
""" + "".join(f"    - NAME: sarc_sdpa_attn_weights_softmax_buffer_half_orin_{c[0]}{n}{'e' if e else ''}\n"
              f"      CACHE: {c}\n      CACHE_N: {n}\n" + ("      CACHE_EXP: true\n" if e else "") for c, n, e in V))
print(" ".join(f"orin_{c[0]}{n}{'e' if e else ''}" for c, n, e in V))
