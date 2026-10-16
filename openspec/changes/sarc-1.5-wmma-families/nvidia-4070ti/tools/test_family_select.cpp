#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <iostream>

using namespace vkcompute::sarc;

int main() {
  DeviceInfo device;
  device.name = "nvidia geforce rtx 4070 ti super";
  device.coopmat = device.int8_coopmat = true;
  device.subgroup_size = 32;
  device.max_shared_bytes = 49152;
  const uint32_t shapes[][2] = {
      {2048, 512}, {2048, 2048}, {2048, 8192}, {3072, 1024}, {3072, 3072},
      {3072, 8192}, {4096, 1024}, {4096, 4096}, {4096, 14336}, {8192, 2048},
      {8192, 3072}, {14336, 4096}};
  for (const Op op : {Op::kQ4gswLinear, Op::kDq8caLinear}) {
    for (const uint32_t m : {2048u, 1280u, 256u, 128u}) {
      for (const auto& kn : shapes) {
        ShapeInfo shape;
        shape.op = op;
        shape.M = m;
        shape.K = kn[0];
        shape.N = kn[1];
        shape.group_size = 128;
        shape.half = shape.io_width_packed = true;
        shape.input = shape.output = Storage::kTexture3D;
        shape.weight = Storage::kTexture2D;
        const auto choice = select(device, shape);
        std::cout << (op == Op::kQ4gswLinear ? "4w" : "8da4w") << " M=" << m
                  << " K=" << kn[0] << " N=" << kn[1] << " -> "
                  << (choice ? choice->kernel_base : "none") << "\n";
      }
    }
  }
}
