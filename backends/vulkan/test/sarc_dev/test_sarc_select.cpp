/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Host-only test of SARC kernel selection (impl/sarc/Select.cpp + the vendor
// tables). No GPU and no Vulkan: it links only the selection sources. For
// every device fixture and Llama prefill shape it compares the table's choice
// with an independent statement of that device's rules, so a table change for
// one device cannot silently change another. Run by sarc/tools/check.sh.
//
// Build (see sarc/tools/check.sh): c++ -std=c++17 -I<parent of repo root>
//   test_sarc_select.cpp impl/sarc/Select.cpp impl/sarc/table_*.cpp
//   [impl/sarc_dev/Overrides.cpp]; args: the sarc yaml files.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

using namespace vkcompute::sarc;

static int failures = 0;
static int checked = 0;

#define EXPECT(cond, ...)                              \
  do {                                                 \
    checked++;                                         \
    if (!(cond)) {                                     \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                        \
      std::printf("\n");                               \
      failures++;                                      \
    }                                                  \
  } while (0)

static DeviceInfo device(
    const char* name,
    bool amd,
    uint32_t sg,
    uint32_t sg_min,
    uint32_t sg_max) {
  DeviceInfo d;
  d.name = name;
  d.is_amd = amd;
  d.subgroup_size = sg;
  d.min_subgroup_size = sg_min;
  d.max_subgroup_size = sg_max;
  d.subgroup_size_control = true;
  d.coopmat = true;
  d.int8_coopmat = true;
  d.max_shared_bytes = 65536;
  return d;
}

static ShapeInfo prefill(int64_t M, int64_t K, int64_t N, Storage io) {
  ShapeInfo s;
  s.op = Op::kQ4gswLinear;
  s.M = M;
  s.K = K;
  s.N = N;
  s.group_size = 128;
  s.half = true;
  s.input = io;
  s.output = io;
  s.weight = Storage::kTexture2D;
  s.io_width_packed = true;
  return s;
}

// (K, N) of every 4w linear in Llama 3.2 1B/3B and 3.1 8B.
static const int64_t kShapes[][2] = {
    {2048, 2048}, {2048, 512},  {2048, 8192},  {8192, 2048}, // 1B
    {3072, 3072}, {3072, 1024}, {3072, 8192},  {8192, 3072}, // 3B
    {4096, 4096}, {4096, 1024}, {4096, 14336}, {14336, 4096}, // 8B
};

static const std::string kQ = "sarc_linear_q4gsw_coopmat_";

// The intended kernel base per device ("" = stock path), restated from the
// 1.4 per-GPU branches independently of the tables.
using Rule = std::function<std::string(const ShapeInfo&)>;

struct Fixture {
  const char* label;
  DeviceInfo dev;
  Rule rule; // 4w
  Rule rule8; // 8da4w
};

static const std::string kD = "sarc_linear_dq8ca_coopmat_";
static std::string none(const ShapeInfo&) {
  return "";
}

static bool active(const DeviceInfo& d, Op op) {
  for (const Row& r : rows()) {
    if (r.op == op &&
        (r.status == Status::kVerified || get_override().allow_unverified) &&
        d.name.find(r.device_substr) != std::string::npos &&
        (r.device_ok == nullptr || r.device_ok(d))) {
      return true;
    }
  }
  return false;
}

int main(int argc, char** argv) {
  const Fixture fixtures[] = {
      {"780M",
       device("amd radeon 780m graphics (radv phoenix)", true, 64, 32, 64),
       [](const ShapeInfo&) { return kQ + "t128x128k32g42s32f32c"; },
       [](const ShapeInfo&) { return kD + "zpg_t128x64k32g42s32"; }},
      {"B580",
       device("intel(r) arc(tm) b580 graphics (bmg g21)", false, 32, 8, 32),
       [](const ShapeInfo&) { return kQ + "t128x128k16g44s16m8fli"; },
       [](const ShapeInfo&) { return kD + "zpg_t256x64k32g48s16m8"; }},
      {"B70",
       device("intel(r) graphics (bmg g31)", false, 32, 8, 32),
       [](const ShapeInfo&) { return kQ + "t128x128k16g44s16m8fli"; },
       [](const ShapeInfo&) { return kD + "zpg_t256x64k32g48s16m8"; }},
      {"4070TiS",
       device("nvidia geforce rtx 4070 ti super", false, 32, 32, 32),
       [](const ShapeInfo& s) {
         if (s.output == Storage::kTexture3D) {
           return kQ +
               (s.M % 256 == 0 && s.N % 128 == 0 && s.N > 512
                    ? "t256x128k16g42s32ga"
                    : "t128x128k16g24s32ga");
         }
         return kQ +
             (s.M % 128 == 0 && s.N % 256 == 0 && s.N > 512
                  ? "t128x256k16g42s32ga"
                  : "t128x128k16g42s32ga");
       },
       [](const ShapeInfo&) { return kD + "zpgtr_t128x128k64g44s32mk32ra"; }},
      {"Orin",
       device("nvidia tegra orin (nvgpu)", false, 32, 32, 32),
       [](const ShapeInfo& s) -> std::string {
         if (s.output != Storage::kTexture3D) {
           return "";
         }
         if (s.K > 8192) {
           return kQ + "t128x128k32g42s32f32";
         }
         if (s.M % 256 == 0 && s.N % 128 == 0 &&
             (s.M != 256 || s.N % 2048 == 0)) {
           return kQ + "t256x128k16g22s32";
         }
         return kQ + "t128x128k16g22s32";
       },
       [](const ShapeInfo& s) -> std::string {
         return s.output == Storage::kTexture3D
             ? kD + "zpgtr_t128x128k64g44s32mk32ra"
             : "";
       }},
      {"M51",
       device("samsung xclipse 940", true, 64, 32, 64),
       [](const ShapeInfo&) { return kQ + "t128x128k16g22s32"; },
       [](const ShapeInfo&) { return kD + "zpgtr_t128x64k32g42s32"; }},
      {"Adreno840",
       device("adreno (tm) 840", false, 64, 64, 128),
       [](const ShapeInfo&) { return kQ + "t64x64k32g21s64m64x32x16"; },
       none},
      {"MaliG1",
       device("mali-g1-ultra", false, 16, 16, 16),
       none,
       none},
      {"GenericNoRows",
       device("some other gpu", false, 32, 32, 32),
       none,
       none},
  };

  for (const Fixture& f : fixtures) {
   for (Op op : {Op::kQ4gswLinear, Op::kDq8caLinear}) {
    const bool on = active(f.dev, op);
    const Rule& rule = op == Op::kQ4gswLinear ? f.rule : f.rule8;
    for (const auto& kn : kShapes) {
      for (Storage io : {Storage::kTexture3D, Storage::kBuffer}) {
        ShapeInfo s = prefill(2048, kn[0], kn[1], io);
        s.op = op;
        const auto c = select(f.dev, s);
        const std::string want = on ? rule(s) : "";
        const std::string got = c.has_value() ? c->kernel_base : "";
        EXPECT(
            got == want,
            "%s K=%lld N=%lld io=%s: got '%s' want '%s'",
            f.label,
            (long long)kn[0],
            (long long)kn[1],
            io == Storage::kBuffer ? "buffer" : "texture3d",
            got.c_str(),
            want.c_str());
        EXPECT(
            builds_on_sarc(f.dev, s) == (get_override().force_path || !want.empty()),
            "%s K=%lld N=%lld: builds_on_sarc mismatch",
            f.label,
            (long long)kn[0],
            (long long)kn[1]);
      }
    }
   }
    // Shapes no SARC variant may take, on every device.
    Override saved = get_override();
    Override all = saved;
    all.allow_unverified = true;
    all.select = nullptr;
    set_override(all);
    ShapeInfo s = prefill(2048, 4096, 4096, Storage::kTexture3D);
    // A row-major dq8ca row must still serve decode / unaligned M when the
    // alignment check is waived (the forced path of pick_sarc_dq8ca_shader).
    {
      ShapeInfo r = s;
      r.op = Op::kDq8caLinear;
      r.int8_layout = Int8Layout::kRowMajor;
      r.M = 1304;
      r.ignore_alignment = true;
      const auto c = select_table(f.dev, r);
      const auto aligned = [&] {
        ShapeInfo a = r;
        a.M = 2048;
        a.ignore_alignment = false;
        return select_table(f.dev, a);
      }();
      if (aligned.has_value() && aligned->rowmajor_a) {
        EXPECT(
            c.has_value() && c->kernel_base == aligned->kernel_base,
            "%s: forced row-major path lost its kernel",
            f.label);
      }
      ShapeInfo w = r; // wrong layout for a row-major row
      w.M = 2048;
      w.ignore_alignment = false;
      w.int8_layout = Int8Layout::k4H4W;
      const auto c4 = select_table(f.dev, w);
      EXPECT(
          !c4.has_value() || !c4->rowmajor_a,
          "%s: 4h4w activations must not get a row-major kernel",
          f.label);
    }
    ShapeInfo u = s;
    u.M = 1304; // unaligned prompt
    ShapeInfo g = s;
    g.M = 1;
    g.gemv = true;
    ShapeInfo b = s;
    b.has_bias = true;
    ShapeInfo h = s;
    h.half = false;
    ShapeInfo mixed = s;
    mixed.input = Storage::kBuffer;
    for (const ShapeInfo* x : {&u, &g, &b, &h, &mixed}) {
      EXPECT(!select(f.dev, *x).has_value(), "%s: must fall back", f.label);
    }
    set_override(saved);
  }

  // SDPA prefill rows: only the AMD wave64 devices (780M, M51); the QK^T
  // (M=S, N=context, K=D) and attn*V (M=S, N=D, K=context) tiles.
  for (const Fixture& f : fixtures) {
    const bool amd = f.dev.is_amd;
    for (Op op : {Op::kSdpaQk, Op::kSdpaAv}) {
      const bool on = amd && active(f.dev, op);
      ShapeInfo s;
      s.op = op;
      s.M = 2048;
      s.N = op == Op::kSdpaQk ? 2048 : 64;
      s.K = op == Op::kSdpaQk ? 64 : 2048;
      s.group_size = s.K;
      s.half = true;
      s.input = s.output = s.weight = Storage::kBuffer;
      const auto c = select(f.dev, s);
      const std::string want = !on ? ""
          : op == Op::kSdpaQk ? "sarc_sdpa_qk_coopmat_t128x64k32g22s64"
                              : "sarc_sdpa_av_coopmat_t64x64k32g22s64";
      EXPECT(
          (c.has_value() ? c->kernel_base : "") == want,
          "%s SDPA op %d: wrong choice",
          f.label,
          (int)op);
      ShapeInfo g = s; // decode stays upstream
      g.M = 1;
      g.gemv = true;
      EXPECT(!select(f.dev, g).has_value(), "%s SDPA decode", f.label);
      ShapeInfo t = s; // texture storage stays upstream
      t.input = t.output = Storage::kTexture3D;
      EXPECT(!select(f.dev, t).has_value(), "%s SDPA texture", f.label);
    }
  }

  // Every kernel name the tables can produce must exist in a yaml.
  if (argc > 1) {
    std::string yamls;
    for (int i = 1; i < argc; i++) {
      std::ifstream in(argv[i]);
      std::stringstream ss;
      ss << in.rdbuf();
      yamls += ss.str();
    }
    const std::pair<uint8_t, const char*> combos[] = {
        {kTex3dTex2d, "_texture3d_texture2d_half"},
        {kBufTex2d, "_buffer_texture2d_half"},
        {kBufBuf, "_buffer_buffer_half"}};
    for (const auto* store : {&rows(), &candidates()}) {
      for (const Row& r : *store) {
        for (const auto& c : combos) {
          if (r.storages & c.first) {
            const std::string n = std::string(r.kernel_base) + c.second;
            // Spelled out, or generated from the base name by the yaml's
            // generate_variant_forall (storage/dtype suffixes).
            EXPECT(
                yamls.find("NAME: " + n + "\n") != std::string::npos ||
                    yamls.find("NAME: " + std::string(r.kernel_base) + "\n") !=
                        std::string::npos,
                "no yaml variant %s",
                n.c_str());
          }
        }
      }
    }
  }

  std::printf(
      "test_sarc_select: %s (%d checks, %zu rows, %zu candidates, dev zone "
      "%s, unverified %s)\n",
      failures == 0 ? "PASS" : "FAIL",
      checked,
      rows().size(),
      candidates().size(),
      get_override().select != nullptr ? "linked" : "absent",
      get_override().allow_unverified ? "on" : "off");
  return failures == 0 ? 0 : 1;
}
