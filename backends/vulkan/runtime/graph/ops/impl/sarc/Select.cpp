/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {

namespace {

std::vector<Row>& row_store() {
  static std::vector<Row> store;
  return store;
}

std::vector<Row>& candidate_store() {
  static std::vector<Row> store;
  return store;
}

Override& override_store() {
  static Override o;
  return o;
}

bool row_matches_device(const Row& row, const DeviceInfo& device) {
  if (device.name.find(row.device_substr) == std::string::npos) {
    return false;
  }
  return row.device_ok == nullptr || row.device_ok(device);
}

bool row_active(const Row& row) {
  return row.status == Status::kVerified || get_override().allow_unverified;
}

} // namespace

void register_rows(const Row* rows, size_t count) {
  row_store().insert(row_store().end(), rows, rows + count);
}

void register_candidates(const Row* rows, size_t count) {
  candidate_store().insert(candidate_store().end(), rows, rows + count);
}

const std::vector<Row>& rows() {
  return row_store();
}

const std::vector<Row>& candidates() {
  return candidate_store();
}

void set_override(const Override& o) {
  override_store() = o;
}

const Override& get_override() {
  return override_store();
}

bool device_has_rows(const DeviceInfo& device, Op op) {
  if (get_override().force_path) {
    return true;
  }
  for (const Row& row : rows()) {
    if (row.op == op && row_active(row) && row_matches_device(row, device)) {
      return true;
    }
  }
  return false;
}

bool q4gsw_coopmat_fits(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const TileDims& dims,
    bool allow_texture_io) {
  // The shaders build only HAS_BIAS=false variants and read no leading dim.
  if (shape.has_bias || shape.gemv || shape.batch != 1 || !shape.half) {
    return false;
  }
  if (!device.coopmat) {
    return false;
  }
  if (shape.weight != Storage::kBuffer && shape.weight != Storage::kTexture2D) {
    return false;
  }
  // One IO_STORAGE parameter covers input and output.
  if (shape.output == Storage::kBuffer) {
    if (shape.input != Storage::kBuffer) {
      return false;
    }
  } else if (shape.output == Storage::kTexture3D) {
    if (!allow_texture_io || shape.input != Storage::kTexture3D ||
        !shape.io_width_packed) {
      return false;
    }
    // The texture epilogue stages SG_GRID_Y * MMA_M rows x WG_TILE_N fp16,
    // on top of Ash/Bsh unless CSH_IN_ASH reuses Ash. A tile over the limit
    // hung a GPU instead of failing pipeline creation (1.4, 2026-08-09).
    if (!dims.csh_in_ash) {
      const uint64_t csh_bytes =
          uint64_t(dims.sg_grid_y) * dims.mma_m * dims.n * 2u;
      if (csh_bytes >= device.max_shared_bytes) {
        return false;
      }
    }
  } else {
    return false;
  }
  return shape.M % dims.m == 0 && shape.N % dims.n == 0 &&
      shape.K % dims.k == 0 && shape.group_size % dims.k == 0;
}

std::optional<Choice> select(const DeviceInfo& device, const ShapeInfo& shape) {
  std::optional<Choice> choice;
  for (const Row& row : rows()) {
    if (row.op != shape.op || !row_active(row) ||
        !row_matches_device(row, device)) {
      continue;
    }
    if (q4gsw_coopmat_fits(device, shape, row.dims, row.allow_texture_io)) {
      choice = Choice{row.kernel_base, row.dims};
      break;
    }
  }
  const Override& o = get_override();
  if (o.select != nullptr) {
    return o.select(device, shape, choice);
  }
  return choice;
}

std::optional<TileDims> dims_for_kernel(const std::string& kernel_name) {
  for (const auto* store : {&rows(), &candidates()}) {
    for (const Row& row : *store) {
      const std::string base(row.kernel_base);
      if (kernel_name.size() > base.size() &&
          kernel_name.compare(0, base.size(), base) == 0 &&
          kernel_name[base.size()] == '_') {
        return row.dims;
      }
    }
  }
  return std::nullopt;
}

} // namespace sarc
} // namespace vkcompute
