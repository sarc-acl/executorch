/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Host-only test of SARC kernel selection (impl/sarc/Select.cpp + the vendor
// tables). No GPU and no Vulkan: it links only the selection sources. It pins
// which kernel every device fixture gets for the Llama prefill shapes, so a
// table change for one device cannot silently change another. Run by
// sarc/tools/check.sh.
//
// Build (see sarc/tools/check.sh): c++ -std=c++17 -I<parent of repo root>
//   test_sarc_select.cpp impl/sarc/Select.cpp impl/sarc/table_*.cpp
//   [impl/sarc_dev/Overrides.cpp]; args: the sarc yaml files.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace vkcompute::sarc;

static int failures = 0;

#define EXPECT(cond, ...)                          \
  do {                                             \
    if (!(cond)) {                                 \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                    \
      std::printf("\n");                           \
      failures++;                                  \
    }                                              \
  } while (0)

static DeviceInfo radeon_780m() {
  DeviceInfo d;
  d.name = "amd radeon 780m graphics (radv phoenix)";
  d.is_amd = true;
  d.subgroup_size = 64;
  d.coopmat = true;
  d.max_shared_bytes = 65536;
  return d;
}

static DeviceInfo arc_b580() {
  DeviceInfo d;
  d.name = "intel(r) arc(tm) b580 graphics (bmg g21)";
  d.subgroup_size = 32;
  d.coopmat = true;
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

static bool verified_row_for(const DeviceInfo& d) {
  for (const Row& r : rows()) {
    if (r.status == Status::kVerified &&
        d.name.find(r.device_substr) != std::string::npos &&
        (r.device_ok == nullptr || r.device_ok(d))) {
      return true;
    }
  }
  return false;
}

int main(int argc, char** argv) {
  const bool dev_zone = get_override().select != nullptr;
  // With the dev zone linked, unverified rows count only when
  // ET_VK_SARC_UNVERIFIED is set; the release build ignores them.
  const DeviceInfo r780m = radeon_780m();
  const bool r780m_active = verified_row_for(r780m) ||
      (dev_zone && get_override().allow_unverified);

  for (const auto& kn : kShapes) {
    for (Storage io : {Storage::kTexture3D, Storage::kBuffer}) {
      const ShapeInfo s = prefill(2048, kn[0], kn[1], io);
      const auto c = select(r780m, s);
      if (r780m_active) {
        EXPECT(
            c.has_value() &&
                c->kernel_base ==
                    "sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c",
            "780M K=%lld N=%lld io=%d: wrong or no choice",
            (long long)kn[0], (long long)kn[1], (int)io);
        if (c.has_value()) {
          EXPECT(c->dims.wg_size() == 256, "780M wg_size %u", c->dims.wg_size());
        }
      } else {
        EXPECT(!c.has_value(), "780M row inactive but chosen");
      }
      // Devices without rows keep the upstream path.
      EXPECT(!select(arc_b580(), s).has_value(), "B580 must have no choice");
    }
  }
  EXPECT(device_has_rows(r780m, Op::kQ4gswLinear) == r780m_active,
         "780M device_has_rows");

  // Shapes the variants must never take.
  {
    DeviceInfo d = r780m;
    Override saved = get_override();
    Override all = saved;
    all.allow_unverified = true;
    all.select = nullptr;
    set_override(all);
    ShapeInfo s = prefill(2048, 2048, 2048, Storage::kTexture3D);
    EXPECT(select(d, s).has_value(), "aligned prefill must be chosen");
    ShapeInfo u = s;
    u.M = 1304; // unaligned prompt
    EXPECT(!select(d, u).has_value(), "unaligned M must fall back");
    ShapeInfo g = s;
    g.M = 1;
    g.gemv = true;
    EXPECT(!select(d, g).has_value(), "decode must fall back");
    ShapeInfo b = s;
    b.has_bias = true;
    EXPECT(!select(d, b).has_value(), "bias must fall back");
    ShapeInfo f = s;
    f.half = false;
    EXPECT(!select(d, f).has_value(), "fp32 must fall back");
    ShapeInfo mixed = s;
    mixed.input = Storage::kBuffer;
    EXPECT(!select(d, mixed).has_value(), "mixed IO storage must fall back");
    set_override(saved);
  }

  // Every kernel name the tables can produce must exist in a yaml. argv[1..]
  // are the yaml files (passed by check.sh).
  std::string yamls;
  for (int i = 1; i < argc; i++) {
    std::ifstream f(argv[i]);
    std::stringstream ss;
    ss << f.rdbuf();
    yamls += ss.str();
  }
  if (argc > 1) {
    for (const auto* store : {&rows(), &candidates()}) {
      for (const Row& r : *store) {
        EXPECT(
            yamls.find(std::string(r.kernel_base) + "_") != std::string::npos,
            "no yaml variant for %s", r.kernel_base);
      }
    }
  }

  std::printf(
      "test_sarc_select: %s (%zu rows, %zu candidates, dev zone %s, "
      "780M row %s)\n",
      failures == 0 ? "PASS" : "FAIL",
      rows().size(),
      candidates().size(),
      dev_zone ? "linked" : "absent",
      r780m_active ? "active" : "inactive");
  return failures == 0 ? 0 : 1;
}
