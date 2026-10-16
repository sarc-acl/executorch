#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <iostream>

using namespace vkcompute::sarc;

namespace vkcompute::sarc {
const char* sdpa_fused_variants_b580();
}

int main() {
  DeviceInfo device;
  device.name = "intel(r) arc(tm) b580 graphics (bmg g21)";
  device.coopmat = true;
  device.int8_coopmat = true;
  device.subgroup_size = 32;
  device.subgroup_size_control = true;
  device.min_subgroup_size = 16;
  device.max_subgroup_size = 32;
  device.max_shared_bytes = 49152;
  const int64_t shapes[][2] = {
      {2048, 2048}, {2048, 512}, {2048, 8192}, {8192, 2048},
      {3072, 3072}, {3072, 1024}, {3072, 8192}, {8192, 3072},
      {4096, 4096}, {4096, 1024}, {4096, 14336}, {14336, 4096}};
  std::cout << "scheme,io,M,K,N,kernel_base\n";
  for (Op op : {Op::kQ4gswLinear, Op::kDq8caLinear}) {
    for (Storage storage : {Storage::kTexture3D, Storage::kBuffer}) {
      for (const auto& dims : shapes) {
        ShapeInfo shape;
        shape.op = op;
        shape.M = 2048;
        shape.K = dims[0];
        shape.N = dims[1];
        shape.group_size = 128;
        shape.half = true;
        shape.input = storage;
        shape.output = storage;
        shape.weight = Storage::kTexture2D;
        shape.io_width_packed = true;
        const auto choice = select(device, shape);
        if (!choice) {
          return 1;
        }
        std::cout << (op == Op::kQ4gswLinear ? "4w" : "8da4w") << ','
                  << (storage == Storage::kTexture3D ? "texture3d" : "buffer")
                  << ',' << shape.M << ',' << shape.K << ',' << shape.N << ','
                  << choice->kernel_base << '\n';
      }
    }
  }
  for (int64_t head : {64, 128}) {
    for (Op op : {Op::kSdpaQk, Op::kSdpaAv}) {
      ShapeInfo shape;
      shape.op = op;
      shape.M = 2048;
      shape.N = op == Op::kSdpaQk ? 2048 : head;
      shape.K = op == Op::kSdpaQk ? head : 2048;
      shape.group_size = 128;
      shape.half = true;
      shape.input = Storage::kBuffer;
      shape.output = Storage::kBuffer;
      shape.weight = Storage::kBuffer;
      const auto choice = select(device, shape);
      if (!choice) {
        return 2;
      }
      std::cout << (op == Op::kSdpaQk ? "qk" : "av") << ",buffer,"
                << shape.M << ',' << shape.K << ',' << shape.N << ','
                << choice->kernel_base << '\n';
    }
  }
  std::cerr << "softmax="
            << (get_override().softmax_variant == nullptr
                    ? "default"
                    : get_override().softmax_variant)
            << '\n';
  std::cerr << "fused=" << sdpa_fused_variants_b580() << '\n';
}
