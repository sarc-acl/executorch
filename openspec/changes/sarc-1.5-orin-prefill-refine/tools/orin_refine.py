"""orin_refine.py: the orin-* candidate profiles (read by gen_orin_sdpa.py). (op, token, predicate or None);
per shape the first fitting entry wins, a shape no entry fits keeps the table's choice.

orin-refine1      SDPA prefill kernels, the best per head_dim of SDPA screens 1 and 2 (results/orin/screens/):
                  QK^T    packed staging, K = 64 per chunk, 128 x 64 tile (all head dims);
                  attn*V  head_dim 128: multi-pass staging 64 x 128; head_dim 64: the 64 x 64 tile (the 128-column
                          tile does not fit head_dim 64 and falls through to it).
                  Needs ET_VK_SARC_UNVERIFIED=1 (the SDPA base rows of OrinSdpa.cpp are kUnverified).
orin-lin-refine2  8da4w linear only: whole-texel weight staging on a 256-thread 128 x 128 tile (8da4w screen 2).
                  An orin-lin-* profile does not enable the SDPA rows.
orin-refine3      orin-refine1 + orin-lin-refine2."""
SDPA1 = [("kSdpaQk", "4070ti_pk_t128x64k64g42s32nf", None),
         ("kSdpaAv", "4070ti_ml_t64x128k32g42s32", None),
         ("kSdpaAv", "4070ti_t64x64k32g42s32", None)]
LIN2 = [("kDq8caLinear", "orin_bf_t128x128k64g24s32mk32ra", None)]
REFINE = {"orin-refine1": SDPA1, "orin-lin-refine2": LIN2, "orin-refine3": SDPA1 + LIN2}
