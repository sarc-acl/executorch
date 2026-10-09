#!/usr/bin/env python3
# gen_rx7600_q4_body.py <release 4w body> <out body>: release 4w body (glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh) ->
# glsl/sarc_dev/sarc_dev_rx7600_q4_body.glslh (round 2, candidate 2). Every edit asserts that it matched exactly once.
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:70])
    t = t.replace(old, new)

rep("// Body of the SARC q4gsw coopmat linear. Included by a template header", """// SARC development zone, RX 7600 campaign round 2 (openspec/changes/sarc-1.5-rx7600-prefill-refine): a copy of
// glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh (the release body is untouched) with the options below, all default off (a variant
// with none of them computes exactly what the release body computes):
//   BT_PAD_H (constant from the wrapper; default 8): padding in fp16 of one row (one N column) of the column-major B staging
//     (B_COLMAJOR), so that the row pitch is WG_TILE_K + BT_PAD_H fp16 (80 bytes for the default; 72 or 88 spread the 16 lanes of a
//     ds_read_b64 fragment load over the LDS banks).
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design).
// Body of the SARC q4gsw coopmat linear. Included by a template header""")
rep("SARC_LINEAR_Q4GSW_COOPMAT_BODY_GLSLH", "SARC_DEV_RX7600_Q4GSW_BODY_GLSLH", 3)
rep("const uint BT_STRIDE_H = WG_TILE_K + 8u;", "const uint BT_STRIDE_H = WG_TILE_K + BT_PAD_H;")
open(dst, "w").write(t)
print("ok", len(t.splitlines()), "lines")
