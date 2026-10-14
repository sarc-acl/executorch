#!/usr/bin/env python3
"""Linear profiles of the RTX 4070 Ti SUPER campaign (dev zone only, no hook needed).

usage: gen_4070ti_profiles.py <executorch tree>      (after gen_4070ti_lin.py and gen_4070ti_bh.py)

  impl/sarc_dev/Overrides.cpp   `// >>> 4070ti lin-preferences` and `// >>> 4070ti lin-profiles`

  4070ti-refine2   8da4w: zpgtr with half-texel weight staging on the shipped tile (bh_t128x128k64g44s32mk32ra)
  4070ti-refine3   refine2 + 4w: the shipped wide tile with the texture3d drain staged in Ash
                   (t256x128k16g42s32gac) where the table uses the wide tile (N > 512)
  4070ti-refine4   the SDPA kernels of 4070ti-refine1 + refine2's 8da4w staging
  4070ti-refine5   refine4 + refine3's 4w tile
The linear variants were within noise at kernel level (results/4070ti/screens); they are gated to close the
linear part of the campaign with measured end-to-end numbers, not because a gain is expected. refine4 and
refine5 put them on top of the SDPA candidate, so that each is measured against its parent.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from devzone import put_block
o = pathlib.Path(sys.argv[1]) / "backends/vulkan/runtime/graph/ops/impl/sarc_dev/Overrides.cpp"
prefs = """// RTX 4070 Ti SUPER linear profiles (tools/gen_4070ti_profiles.py).
bool n_above_512_4070ti(const ShapeInfo& s) {
  return s.N > 512;
}
const Preference k4070tiRefine2[] = {
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
};
const Preference k4070tiRefine3[] = {
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "4070ti_t256x128k16g42s32gac", n_above_512_4070ti},
};
const Preference k4070tiRefine4[] = {
    {Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", head_dim_64_4070ti},
    {Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", head_dim_64_4070ti},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
};
const Preference k4070tiRefine5[] = {
    {Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", head_dim_64_4070ti},
    {Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", head_dim_64_4070ti},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "4070ti_t256x128k16g42s32gac", n_above_512_4070ti},
};
"""
profs = """    {"4070ti-refine2", k4070tiRefine2, sizeof(k4070tiRefine2) / sizeof(Preference)},
    {"4070ti-refine3", k4070tiRefine3, sizeof(k4070tiRefine3) / sizeof(Preference)},
    {"4070ti-refine4", k4070tiRefine4, sizeof(k4070tiRefine4) / sizeof(Preference)},
    {"4070ti-refine5", k4070tiRefine5, sizeof(k4070tiRefine5) / sizeof(Preference)},
"""
put_block(o, "struct Profile {\n", "lin-preferences", prefs)
put_block(o, "};\nconst Profile* requested_profile() {", "lin-profiles", profs, "    ")
print("4070ti linear profiles generated")
