#!/usr/bin/env python3
"""Batch 3 (dev zone only): two more shapes per scheme, a PROF twin of the 8da4w g22 tile, and the
ET_VK_SARC_DEV_PROFILE mechanism in impl/sarc_dev/Overrides.cpp (a named list of preferred sweep tiles with shape
predicates; shapes a preferred tile does not cover keep the table's choice). Idempotent. usage: <executorch tree>"""
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan/runtime/graph/ops"
MARK = "780M prefill refine 2026-10-03, batch 3"
LDS_MAX = 60000
DQ = [("t64x128k32g41s32", 64, 128, 32, 4, 1, 32, 1), ("t256x32k32g14s32", 256, 32, 32, 1, 4, 32, 4)]
y = root / "glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpg_sweep.yaml"; t = y.read_text()
if MARK not in t:
    t += f"    # {MARK}: 4-wave workgroups with the 64x32 per-wave tile in the other two arrangements.\n"
    for tok, m, n, k, sx, sy, sg, ab in DQ:
        wg = sx * sy * sg
        assert (m // 4) * (k // 4) == ab * wg and ((k // 4) * n) % wg == 0, tok
        assert 2 * m * k + 2 * n * k + 8 * m + 12 * n + sy * 16 * n * 2 <= LDS_MAX, tok
        t += f"    - NAME: sarc_linear_dq8ca_coopmat_zpg_sweep_{tok}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n"
        t += f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      A_MAP_FULL: true\n"
        if ab > 1: t += f"      A_MULTI_BLOCK: true\n      A_BLOCKS: {ab}\n"
    y.write_text(t)
Q4 = [("t64x256k32g41s32f32c", 64, 256, 32, 4, 1, 32)]
y = root / "glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.yaml"; t = y.read_text()
if MARK not in t:
    t += f"    # {MARK}: the wide tile with a shallower chunk, and as a 4-wave workgroup.\n"
    for tok, m, n, k, sx, sy, sg in Q4:
        wg = sx * sy * sg
        assert m % (wg // (k // 8)) == 0 and k % (wg // (n // 8)) == 0, tok
        assert 2 * m * ((k + 8) // 8) >= sy * 16 * (n // 8), tok
        assert 4 * (m * (k + 8) + k * (n + 8)) <= LDS_MAX, tok
        t += f"    - NAME: sarc_linear_q4gsw_coopmat_sweep_{tok}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n"
        t += f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {sg}\n      ACC_FP32: true\n      CSH_IN_ASH: true\n"
    y.write_text(t)
y = root / "glsl/sarc_dev/sarc_dev_prof_dq8ca_zpg.yaml"; t = y.read_text()
if "g22s32p" not in t:
    t += """    - NAME: sarc_dev_prof_dq8ca_zpg_t128x64k32g22s32p_texture3d_texture2d_half
      IO_STORAGE: texture3d
      SG_GRID_X: 2
      SG_GRID_Y: 2
      A_MAP_FULL: true
      A_MULTI_BLOCK: true
      A_BLOCKS: 2
"""
    y.write_text(t)
y = root / "glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml"; t = y.read_text()
if "t128x256k32g42s32f32cp" not in t:
    t += """    - NAME: sarc_dev_prof_q4gsw_t128x256k32g42s32f32cp_texture3d_texture2d_half
      IO_STORAGE: texture3d
      WG_TILE_N: 256
      WG_TILE_K: 32
      SG_GRID_X: 4
      SG_GRID_Y: 2
      ACC_FP32: true
      CSH_IN_ASH: true
"""
    y.write_text(t)
o = root / "impl/sarc_dev/Overrides.cpp"; t = o.read_text()
if MARK not in t:
    a = "    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml):"
    assert t.count(a) == 1
    rows = f"    // {MARK}.\n"
    for tok, m, n, k, sx, sy, sg in Q4:
        rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "sarc_linear_q4gsw_coopmat_sweep_{tok}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {sg}, 16, true}}, kTex3dTex2d, nullptr,\n     Status::kUnverified}},\n'
    rows += '    {"", nullptr, Op::kQ4gswLinear,\n     "sarc_dev_prof_q4gsw_t128x256k32g42s32f32cp", tile_mnk(128, 256, 32, 4, 2, true),\n     kTex3dTex2d, nullptr, Status::kUnverified},\n'
    t = t.replace(a, rows + a)
    a = "    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_dq8ca_zpg.yaml)."
    assert t.count(a) == 1
    rows = f"    // {MARK}.\n"
    for tok, m, n, k, sx, sy, sg, ab in DQ:
        rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "sarc_linear_dq8ca_coopmat_zpg_sweep_{tok}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {sg}, 16, false}}, kTex3dTex2d, nullptr,\n     Status::kUnverified}},\n'
    rows += '    {"", nullptr, Op::kDq8caLinear,\n     "sarc_dev_prof_dq8ca_zpg_t128x64k32g22s32p", dq_tile(128, 64, 2, 2),\n     kTex3dTex2d, nullptr, Status::kUnverified},\n'
    t = t.replace(a, rows + a)
    # --- profile mechanism ---
    a = "std::optional<Choice> dev_select(\n    const DeviceInfo& device,\n    const ShapeInfo& shape,\n    const std::optional<Choice>& table_choice) {\n"
    assert t.count(a) == 1
    t = t.replace(a, """// ET_VK_SARC_DEV_PROFILE=<name>: a named set of preferred sweep tiles. A shape
// that a preferred tile covers (the row fits and the entry's predicate holds)
// runs it; every other shape keeps the table's choice. Unlike the *_VARIANT
// variables it never sends a shape to the non-SARC fallback and never builds
// the SARC path on a device without rows.
struct Preference {
  Op op;
  const char* token; // tile token of a candidate or release row
  bool (*shape_ok)(const ShapeInfo&); // or null
};
// The wide 4w tile stages A once per 256 output columns but leaves 2 workgroups
// per WGP (54 KiB LDS); it only pays once the dispatch is at least 4 tiles wide.
bool n_at_least_1024(const ShapeInfo& s) {
  return s.N >= 1024;
}
// 780M, openspec/changes/sarc-1.5-780m-prefill-refine.
const Preference k780mRefine1[] = {
    {Op::kDq8caLinear, "t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
};
const Preference k780mRefine1Dq[] = {
    {Op::kDq8caLinear, "t128x64k32g22s32", nullptr},
};
const Preference k780mRefine1Q4[] = {
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
};
struct Profile {
  const char* name;
  const Preference* prefs;
  size_t count;
};
const Profile kProfiles[] = {
    {"780m-refine1", k780mRefine1, sizeof(k780mRefine1) / sizeof(Preference)},
    {"780m-refine1-dq", k780mRefine1Dq, sizeof(k780mRefine1Dq) / sizeof(Preference)},
    {"780m-refine1-q4", k780mRefine1Q4, sizeof(k780mRefine1Q4) / sizeof(Preference)},
};
const Profile* requested_profile() {
  static const Profile* p = []() -> const Profile* {
    const char* e = std::getenv("ET_VK_SARC_DEV_PROFILE");
    if (e == nullptr || *e == 0) {
      return nullptr;
    }
    for (const Profile& pr : kProfiles) {
      if (std::strcmp(pr.name, e) == 0) {
        return &pr;
      }
    }
    std::cerr << "[sarc_dev] unknown ET_VK_SARC_DEV_PROFILE=" << e << std::endl;
    std::abort();
  }();
  return p;
}

""" + a + """  if (const Profile* profile = requested_profile()) {
    if (table_choice.has_value()) {
      for (size_t i = 0; i < profile->count; i++) {
        const Preference& pref = profile->prefs[i];
        if (pref.op != shape.op ||
            (pref.shape_ok != nullptr && !pref.shape_ok(shape))) {
          continue;
        }
        for (const auto* store : {&candidates(), &rows()}) {
          for (const Row& row : *store) {
            if (row.op == shape.op &&
                ends_with(row.kernel_base, std::string("_") + pref.token) &&
                q4gsw_coopmat_fits(device, shape, row)) {
              return Choice{row.kernel_base, row.dims, row.rowmajor_a};
            }
          }
        }
      }
    }
    return table_choice;
  }
""")
    a = "    if (o.allow_unverified || o.force_path) {"
    assert t.count(a) == 1
    t = t.replace(a, """    if (requested_profile() != nullptr) {
      std::cerr << "[sarc_dev] profile active: " << requested_profile()->name
                << std::endl;
    }
""" + a)
    a = "//   ET_VK_SARC_DQ8CA_VARIANT=<tile>   same for 8da4w prefill"
    assert t.count(a) == 1
    t = t.replace(a, """//   ET_VK_SARC_DEV_PROFILE=<name>     a named set of preferred sweep tiles (see
//                                     kProfiles); shapes they do not cover keep
//                                     the table's choice. Takes precedence over
//                                     every variable above
""" + a)
    o.write_text(t)
print("batch 3 + profile mechanism added")
