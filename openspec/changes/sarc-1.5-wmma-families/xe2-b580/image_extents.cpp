#include <executorch/backends/vulkan/runtime/api/containers/Tensor.h>

#include <iostream>

// These pure functions are defined in the pinned Tensor.cpp, without public declarations.
namespace vkcompute::api {
PackedDimInfo calculate_packed_dim_info(utils::GPUMemoryLayout, utils::StorageType);
std::vector<int64_t> calculate_padded_sizes(
    const std::vector<int64_t>&, const PackedDimInfo&);
std::vector<int64_t> calculate_axis_map(
    const std::vector<int64_t>&, utils::AxisMapLayout);
utils::uvec3 calculate_image_extents(
    vkapi::ScalarType,
    const PackedDimInfo&,
    const std::vector<int64_t>&,
    const std::vector<int64_t>&);
} // namespace vkcompute::api

int main() {
  using namespace vkcompute;
  for (const auto& sizes : std::vector<std::vector<int64_t>>{
           {1, 2048, 14336}, {5120, 32}, {5120, 64}}) {
    const auto packed = api::calculate_packed_dim_info(
        utils::kWidthPacked, utils::kTexture3D);
    const auto padded = api::calculate_padded_sizes(sizes, packed);
    const auto axes = api::calculate_axis_map(sizes, utils::kDefaultAxisMap);
    const auto extents = api::calculate_image_extents(
        vkapi::kHalf, packed, padded, axes);
    for (auto size : sizes) {
      std::cout << size << ' ';
    }
    std::cout << ": " << extents[0] << ' ' << extents[1] << ' '
              << extents[2] << '\n';
  }
}
