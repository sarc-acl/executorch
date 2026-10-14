"""orin_refine.py: the orin-* candidate profiles (read by gen_orin_sdpa.py). (op, token, predicate or None);
per shape the first fitting entry wins, a shape no entry fits keeps the table's choice.

orin-refine1      SDPA prefill kernels, the best per head_dim of SDPA screens 1 and 2 (results/orin/screens/):
                  QK^T    packed staging, K = 64 per chunk, 128 x 64 tile (all head dims);
                  attn*V  head_dim 128: multi-pass staging 64 x 128; head_dim 64: the 64 x 64 tile (the 128-column
                          tile does not fit head_dim 64 and falls through to it).
                  Needs ET_VK_SARC_UNVERIFIED=1 (the SDPA base rows of OrinSdpa.cpp are kUnverified).
orin-lin-refine2  8da4w linear only: whole-texel weight staging on a 256-thread 128 x 128 tile (8da4w screen 2).
                  An orin-lin-* profile does not enable the SDPA rows.
orin-refine3      orin-refine1 + orin-lin-refine2.
orin-lin-refine3  orin-lin-refine2 + the 4w texel-wise weight staging tile on the fp32-accumulating shapes (K > 8192).
orin-refine4      orin-refine1 + orin-lin-refine3: everything. With ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf it is
                  candidate 1h plus the linear changes.
orin-lin-refine5  orin-lin-refine3 + the 4w tile of 4w screen 2 on the shapes the table gives to the 256 x 128
                  tile (orin_256 in impl/sarc/table_nvidia.cpp: K <= 8192, and not the M = 256, N % 2048 != 0 case):
                  the same 256 x 128 K = 16 tile with column-major weight staging on a 4 x 2 subgroup grid.
orin-refine5      orin-refine1 + orin-lin-refine5: everything; the softmax variant is named by the environment."""
SDPA1 = [("kSdpaQk", "4070ti_pk_t128x64k64g42s32nf", None),
         ("kSdpaAv", "4070ti_ml_t64x128k32g42s32", None),
         ("kSdpaAv", "4070ti_t64x64k32g42s32", None)]
LIN2 = [("kDq8caLinear", "orin_bf_t128x128k64g24s32mk32ra", None)]
# 4w screen 1: only on the fp32-accumulating shape (K > 8192, i.e. 8B w2) does another tile beat the shipped one:
# the 780M campaign's texel-wise weight staging on the same 128 x 128 K = 32 tile (1.11x at kernel level).
LIN3 = LIN2 + [("kQ4gswLinear", "bx_t128x128k32g42s32f32c", "k_above_8192_orin")]
# 4w screen 2 (results/orin/screens/screen2-4w.txt): 1.3 to 1.8 % at kernel level on the K <= 8192 shapes.
LIN5 = LIN3 + [("kQ4gswLinear", "orin_t256x128k16g42s32bt", "orin_256_shape")]
REFINE = {"orin-refine1": SDPA1, "orin-lin-refine2": LIN2, "orin-refine3": SDPA1 + LIN2,
          "orin-lin-refine3": LIN3, "orin-refine4": SDPA1 + LIN3,
          "orin-lin-refine5": LIN5, "orin-refine5": SDPA1 + LIN5}
