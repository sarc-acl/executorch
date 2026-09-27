/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC selection rows for AMD GPUs. First matching row wins, so list the most
// specific device first. Status kVerified requires sarc/tools/verify.sh
// evidence on that device (see sarc/README.md).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

// The shipped RDNA coopmat variants were built and validated for wave64
// adapters that support cooperative matrix; they force subgroup 32 per
// pipeline (SUBGROUP_SIZE in the yaml).
bool amd_wave64(const DeviceInfo& d) {
  return d.is_amd && d.subgroup_size == 64;
}

// t128x128k32g42s32 with fp32 accumulate and the drain staged in Ash.
constexpr TileDims k780mDims = {128, 128, 32, 4, 2, 32, 16, true};

const Row kAmdRows[] = {
    // Radeon 780M (gfx1103, RADV, Mesa 25.2.7). Verified on release 1.5
    // (openspec/changes/sarc-1.5-bootstrap/results/780m): 2048-token prefill
    // 1B/3B/8B 1916/683/352 tok/s vs stock 1.5 1205/421/194; kernel times
    // within -1.6..-0.2 % of the 1.4 study; next token == tiled; pdiff pass.
    {"780m",
     amd_wave64,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c",
     k780mDims,
     /*allow_texture_io=*/true,
     Status::kVerified},
};

struct Registrar {
  Registrar() {
    register_rows(kAmdRows, sizeof(kAmdRows) / sizeof(kAmdRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
