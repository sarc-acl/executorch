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
Selected by ET_VK_SARC_SOFTMAX_VARIANT=orin_<variant> through the release hook Override::softmax_variant.

RESULT (SDPA screen 4, results/orin/screens/sdpa-screen4.csv): all of them are slower than 4070ti_nzf. Local
array 1.13x to 1.42x the time; shared memory 1.86x (8 texels per worker, 8 KB), 2.6x (16 KB), 4.6x (32 KB):
the time grows with the shared memory a workgroup declares. The re-reads were not the cost. Kept as measured.

Second family, glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_orin_wg.{glsl,yaml}: what a row costs besides its
traffic is 14 barriers (two tree reductions over 64 workers, 7 each) for, on average, 4 texels per worker.
      sarc_sdpa_attn_weights_softmax_buffer_half_orin_z64        worker 0 walks the same tree alone: 2 barriers
                                                                per reduction; same pairs in the same order as
                                                                4070ti_nzf, so meant to be bit-identical
      ..._orin_t{8,16,32}   the tree as it is, with fewer workers per row
      ..._orin_z{8,16,32}   worker 0 walks the tree, fewer workers
      ..._orin_f{1,2,4,8,16,32,64}   one barrier per reduction: every worker combines the partial results
                                     itself, in index order
The workgroup size is fixed in the shader for these (the node asks for 64 x 1 x 1 by specialization constants,
which a shader without those ids ignores; the dispatch stays one workgroup per row). Every variant but z64
sums the exponentials of a row in another order than 4070ti_nzf (fp32, rounded once on the store): an
arithmetic change, judged by the reference error, not by bit identity.

RESULT so far (SDPA screen 5 on the second Orin, round 1): f64 0.95x the time of 4070ti_nzf; z64 1.76x; and
fewer workers is slower in proportion (32 workers 1.3x, 16 2.1x, 8 3.8x). So the third batch goes the other way:
      ..._orin_f{128,256}          more workers per row, flat combination
      ..._orin_g{32,64,128,256,512}  reduction inside each subgroup (subgroupMax / subgroupAdd, no barrier), one
                                   barrier, then every worker combines the subgroups' results in index order
      ..._orin_xp{0,1,2}           MEASUREMENT ONLY, wrong output, never a candidate: 4070ti_nzf that stops after
                                   the bounds check / after pass 1 / after pass 2 (one texel written so that the
                                   pass is not removed), to see where a row's time goes.

REVISION 2026-10-06 (review finding). As first generated, every lane of a subgroup stored the reduced value into
its subgroup's slot (shared_max[gl_SubgroupID] = subgroupMax(...)): unordered non-atomic writes of one location
by several invocations, a data race by the Vulkan memory model even though the values are equal. Now the
reduction is still computed by every lane and stored by the elected lane alone (subgroupElect()); the workgroup
barriers are unchanged. This changes the SPIR-V of every g variant: the measurements of the g variants up to
build topic13 (SDPA screens 6 and 7, sdpa-error5, sessions s6-c4 and s7-final, probe/final-g64) are of the first
form and are kept as measured; candidate 4 (orin_g64) is gated and measured again on the corrected build."""
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

# ---- second family: workers per row and the form of the two reductions
s = (g / "sarc_dev/sarc_sdpa_attn_weights_softmax_4070ti.glsl").read_text()
s = s[s.index("#version 450 core"):]
s = sub(s, "#define NUM_WORKERS_PER_WG 64\n", "#define NUM_WORKERS_PER_WG ${WORKERS}\n")
s = sub(s, "#extension GL_EXT_control_flow_attributes : require\n",
        "#extension GL_EXT_control_flow_attributes : require\n$if RED == \"subgroup\":\n"
        "  #extension GL_KHR_shader_subgroup_basic : require\n  #extension GL_KHR_shader_subgroup_arithmetic : require\n")
for arr, loc, op in (("shared_max", "local_max", "subgroupMax"), ("shared_exp_sum", "local_exp_sum", "subgroupAdd")):
    s = sub(s, f"  {arr}[worker_id] = {loc};\n",
            f"$if RED == \"subgroup\":\n    // orin g: reduced inside the subgroup by every lane; one slot per subgroup, written by the\n"
            f"    // elected lane alone (every lane storing the same value is still a data race).\n"
            f"    const SOFTMAX_ACC_T {arr}_sg = {op}({loc});\n"
            f"    if (subgroupElect()) {{\n      {arr}[gl_SubgroupID] = {arr}_sg;\n    }}\n$else:\n    {arr}[worker_id] = {loc};\n")
XP = ("    // orin xp: MEASUREMENT ONLY (wrong output): the kernel ends here.\n"
      "    if (worker_id == 0) {{\n      store_attn_weights_softmax_c4(\n"
      "          VEC4_T(T({v})), 0, s, q_h, context_texel_len, attn_S, Q_H);\n    }}\n    return;\n")
B = "  // =========================================================================\n  // Pass "
s = sub(s, B + "1:", "$if PHASE == 0:\n" + XP.format(v="1.0") + B + "1:")
s = sub(s, B + "2:", "$if PHASE == 1:\n" + XP.format(v="global_max") + B + "2:")
s = sub(s, B + "3:", "$if PHASE == 2:\n" + XP.format(v="local_exp_sum") + B + "3:")
s = sub(s, "layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n",
        "// orin: the workgroup is one row's workers; fixed here, the node's specialization constants are not used.\n"
        "layout(local_size_x = NUM_WORKERS_PER_WG, local_size_y = 1, local_size_z = 1) in;\n")
def red(what, arr, comb, result):
    tree = f"""  // Tree reduction to {what}
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {{
    if (worker_id < i) {{
      {arr}[worker_id] = {comb('%s[worker_id]' % arr, '%s[worker_id + i]' % arr, 10)};
    }}
    memoryBarrierShared();
    barrier();
  }}

  {result} = {arr}[0];
"""
    return tree, f"""$if RED == "tree":
{tree.rstrip(chr(10)).replace(chr(10), chr(10) + "  ").replace("  " + chr(10), chr(10))}
$elif RED == "serial":
    // orin z: worker 0 walks the same tree alone (same pairs, same order), one barrier after it.
    if (worker_id == 0) {{
      for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {{
        for (int k = 0; k < i; ++k) {{
          {arr}[k] = {comb('%s[k]' % arr, '%s[k + i]' % arr, 14)};
        }}
      }}
    }}
    memoryBarrierShared();
    barrier();

    {result} = {arr}[0];
$elif RED == "subgroup":
    // orin g: every worker combines the subgroups' results, in subgroup order.
    SOFTMAX_ACC_T {arr}_all = {arr}[0];
    for (uint k = 1u; k < gl_NumSubgroups; ++k) {{
      {arr}_all = {comb('%s_all' % arr, '%s[k]' % arr, 10)};
    }}
    {result} = {arr}_all;
$else:
    // orin f: no second barrier; every worker combines the partial results itself, in index order.
    SOFTMAX_ACC_T {arr}_all = {arr}[0];
    for (int k = 1; k < NUM_WORKERS_PER_WG; ++k) {{
      {arr}_all = {comb('%s_all' % arr, '%s[k]' % arr, 10)};
    }}
    {result} = {arr}_all;
"""
mx = lambda a, b, ind: f"max(\n{' ' * ind}{a}, {b})"
ad = lambda a, b, ind: f"{a} +\n{' ' * ind}{b}"
for what, arr, comb, result in (("find the global max", "shared_max", mx, "const SOFTMAX_ACC_T global_max"),
                                ("compute the overall exp sum", "shared_exp_sum", ad, "local_exp_sum")):
    old, new = red(what, arr, comb, result); s = sub(s, old, new)
F2 = "sarc_sdpa_attn_weights_softmax_orin_wg"
W = [("tree", "t", n) for n in (8, 16, 32)] + [("serial", "z", n) for n in (8, 16, 32, 64)] + [("flat", "f", n) for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)] + [("subgroup", "g", n) for n in (32, 64, 128, 256, 512)]
write_new(g / f"sarc_dev/{F2}.glsl", HDR.replace("in which a\n * worker keeps the first CACHE_N texels of its row after pass 1 (local array or shared memory) instead of\n * loading them again in passes 2 and 3, and optionally (CACHE_EXP) keeps exp(x - max) after pass 2.\n * Same values, operations and order as the 4070ti variant with the same NZ / ACC32.",
    "with WORKERS workers\n * per row (the workgroup size is fixed here) and the two reductions as a barrier tree (tree), as the same tree\n * walked by worker 0 alone (serial), or combined by every worker in index order after one barrier (flat).\n * serial with 64 workers has the arithmetic of the 4070ti variant; the others sum a row in another order.") + s)
assert "WORKERS workers" in (g / f"sarc_dev/{F2}.glsl").read_text()
write_new(g / f"sarc_dev/{F2}.yaml", f"""# SARC development zone, Jetson Orin: LLM softmax variants with fewer barriers per row (generated by gen_orin_softmax.py). Not shipped.

{F2}:
  parameter_names_with_default_values:
    IN_DTYPE: half
    OUT_DTYPE: half
    STORAGE: buffer
    MODE: llm
    NZ: true
    ACC32: true
    WORKERS: 64
    RED: serial
    PHASE: 3
  shader_variants:
""" + "".join(f"    - NAME: sarc_sdpa_attn_weights_softmax_buffer_half_orin_{c}{n}\n      WORKERS: {n}\n      RED: {r}\n" for r, c, n in W)
    + "".join(f"    - NAME: sarc_sdpa_attn_weights_softmax_buffer_half_orin_xp{n}\n      RED: tree\n      PHASE: {n}\n" for n in (0, 1, 2)))
print(" ".join(f"orin_{c}{n}" for r, c, n in W), "orin_xp0 orin_xp1 orin_xp2")
