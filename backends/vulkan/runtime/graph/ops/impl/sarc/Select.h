/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// SARC kernel selection: which SARC shader variant (if any) runs a given op on
// a given device and shape. Pure C++ (no Vulkan / ComputeGraph types) so it can
// be unit-tested on a host without a GPU (backends/vulkan/test/sarc_dev/).
//
// Rows live in per-vendor tables (table_<vendor>.cpp) and register themselves
// at static-init time. Each row is verified (validated on its device, shipped)
// or unverified (inert unless the development zone enables it). The
// development zone can also install an override (sweep variants, forced
// fallback); a release build has no override.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vkcompute {
namespace sarc {

enum class Op {
  kQ4gswLinear, // 4w: fp16 activations, 4-bit group-symmetric weights
  kDq8caLinear, // 8da4w: dynamic int8 activations, 4-bit weights
};

// Layout of the int8 activations of kDq8caLinear. The zpg kernels (and the
// release-1.5 fallbacks) read 4h4w blocks; the zpgtr kernels read row-major
// kPackedInt8_4W, which only they can read.
enum class Int8Layout { kAny, k4H4W, kRowMajor };

enum class Storage { kBuffer, kTexture2D, kTexture3D, kOther };

enum class Status { kVerified, kUnverified };

struct DeviceInfo {
  std::string name; // lower-case
  bool is_amd = false;
  uint32_t subgroup_size = 0; // default subgroup size
  uint32_t min_subgroup_size = 0;
  uint32_t max_subgroup_size = 0;
  bool subgroup_size_control = false; // VK_EXT_subgroup_size_control
  bool coopmat = false;
  bool int8_coopmat = false; // an int8 cooperative-matrix shape is enumerated
  uint32_t max_shared_bytes = 0;
};

struct ShapeInfo {
  Op op = Op::kQ4gswLinear;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t group_size = 0;
  int64_t batch = 1; // product of leading dims
  bool gemv = false;
  bool has_bias = false;
  bool half = false;
  Storage input = Storage::kOther;
  Storage output = Storage::kOther;
  Storage weight = Storage::kOther;
  bool io_width_packed = false;
  Int8Layout int8_layout = Int8Layout::kAny; // kDq8caLinear only
  // A row-major 8da4w op must stay on its kernel for every M (nothing else
  // reads that layout): skip the tile-alignment check; the grid rounds up.
  bool ignore_alignment = false;
};

struct TileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t sg_grid_x;
  uint32_t sg_grid_y;
  uint32_t subgroup_size;
  uint32_t mma_m;
  bool csh_in_ash; // texture3d drain staged in Ash (no extra Csh LDS)
  uint32_t wg_size() const {
    return sg_grid_x * sg_grid_y * subgroup_size;
  }
};

// Storage combinations (IO_STORAGE x WEIGHT_STORAGE) a variant is built for;
// the yaml must contain <kernel_base>_<io>_<weight>_half for each set bit.
enum StorageSet : uint8_t {
  kTex3dTex2d = 1u << 0, // texture3d input/output, texture2d weight
  kBufTex2d = 1u << 1, // buffer input/output, texture2d weight
  kBufBuf = 1u << 2, // buffer input/output, buffer weight
};

struct Row {
  const char* device_substr; // matched against DeviceInfo::name
  bool (*device_ok)(const DeviceInfo&); // extra device requirements, or null
  Op op;
  const char* kernel_base; // shader name without _<io>_<weight>_<dtype>
  TileDims dims;
  uint8_t storages; // StorageSet bits
  // Extra shape requirement (beyond tile alignment), or null. Rows are tried
  // in order, so a later row can serve the shapes an earlier one rejects.
  bool (*shape_ok)(const ShapeInfo&);
  Status status;
  bool rowmajor_a = false; // kDq8caLinear: reads row-major int8 activations
};

struct Choice {
  std::string kernel_base;
  TileDims dims;
  bool rowmajor_a = false;
};

// Registers rows; call from a static initializer in a table_*.cpp file.
void register_rows(const Row* rows, size_t count);

// Rows that are only ever selected by name (development sweep candidates).
void register_candidates(const Row* rows, size_t count);

// All registered rows, for tests and tools.
const std::vector<Row>& rows();
const std::vector<Row>& candidates();

// Development-zone hook. Given the table's choice, return the choice to use
// (std::nullopt makes the op use its non-SARC fallback kernel).
struct Override {
  bool allow_unverified = false;
  // Build ops on the SARC path even on a device without rows (sweeps).
  bool force_path = false;
  std::optional<Choice> (*select)(
      const DeviceInfo& device,
      const ShapeInfo& shape,
      const std::optional<Choice>& table_choice) = nullptr;
};
void set_override(const Override& o);
const Override& get_override();


// The shape/device constraints of `row` (storage combination, alignment,
// subgroup size, shared memory, int8 layout, the row's shape predicate).
bool q4gsw_coopmat_fits(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const Row& row);

// Table lookup + fit check, then the override (if any).
std::optional<Choice> select(const DeviceInfo& device, const ShapeInfo& shape);

// Table lookup + fit check only (no override).
std::optional<Choice> select_table(
    const DeviceInfo& device,
    const ShapeInfo& shape);

// Whether an op whose build-time shape is `shape` is built on the SARC path:
// a row applies to it (or the dev override forces the path). Otherwise the op
// keeps the upstream path entirely, including for later resizes.
bool builds_on_sarc(const DeviceInfo& device, const ShapeInfo& shape);

// Tile dims for a SARC kernel name (release or candidate row), by base-name
// prefix. Used for launch geometry; no name parsing.
std::optional<TileDims> dims_for_kernel(const std::string& kernel_name);

} // namespace sarc
} // namespace vkcompute
