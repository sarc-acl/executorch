#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdio>

using namespace vkcompute::sarc;

int main() {
  DeviceInfo device;
  device.name = "tegra orin";
  device.coopmat = true;
  device.int8_coopmat = true;
  device.subgroup_size = 32;
  device.min_subgroup_size = 32;
  device.max_subgroup_size = 32;
  device.subgroup_size_control = true;
  device.max_shared_bytes = 49152;
  const int64_t nk[][2] = {
      {512, 2048}, {2048, 2048}, {8192, 2048}, {2048, 8192},
      {1024, 3072}, {3072, 3072}, {8192, 3072}, {3072, 8192},
      {1024, 4096}, {4096, 4096}, {14336, 4096}, {4096, 14336}};
  std::puts("op,M,N,K,storage,weight,ignore_alignment,kernel,softmax");
  for (const Op op : {Op::kQ4gswLinear, Op::kDq8caLinear}) {
    for (const auto& shape : nk) {
      for (const int64_t m : {1, 128, 256, 1972, 2048, 2560}) {
        for (const int storage : {0, 1, 2}) {
          for (const bool ignore : {false, true}) {
            ShapeInfo s;
            s.op = op;
            s.M = m;
            s.N = shape[0];
            s.K = shape[1];
            s.group_size = 128;
            s.half = true;
            s.gemv = m == 1;
            s.input = s.output = storage == 0 ? Storage::kTexture3D : Storage::kBuffer;
            s.weight = storage == 2 ? Storage::kBuffer : Storage::kTexture2D;
            s.io_width_packed = storage == 0;
            s.ignore_alignment = ignore;
            const auto choice = select(device, s);
            std::printf("%d,%lld,%lld,%lld,%d,%d,%d,%s,%s\n",
                        static_cast<int>(op), static_cast<long long>(m),
                        static_cast<long long>(s.N), static_cast<long long>(s.K),
                        storage, static_cast<int>(s.weight), ignore,
                        choice ? choice->kernel_base.c_str() : "NONE",
                        get_override().softmax_variant ? get_override().softmax_variant : "NONE");
          }
        }
      }
    }
  }
  for (const Op op : {Op::kSdpaQk, Op::kSdpaAv}) {
    for (const int64_t head_dim : {64, 128}) {
      ShapeInfo s;
      s.op = op;
      s.M = 2048;
      s.N = op == Op::kSdpaQk ? 2560 : head_dim;
      s.K = op == Op::kSdpaQk ? head_dim : 2560;
      s.half = true;
      s.input = s.output = s.weight = Storage::kBuffer;
      const auto choice = select(device, s);
      std::printf("%d,2048,%lld,%lld,2,0,0,%s,%s\n",
                  static_cast<int>(op), static_cast<long long>(s.N),
                  static_cast<long long>(s.K), choice ? choice->kernel_base.c_str() : "NONE",
                  get_override().softmax_variant ? get_override().softmax_variant : "NONE");
    }
  }
  return 0;
}
