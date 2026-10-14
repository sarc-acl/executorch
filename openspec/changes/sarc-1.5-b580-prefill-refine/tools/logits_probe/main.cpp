// logits_probe (campaign-local copy of sarc-1.5-e2e-benchmark/kit/logits_probe with a window and a full dump):
// run one prefill of a Llama .pte (Vulkan delegate) on token ids [offset, offset + length) of an id file and
// write the last-position logits: top-10 as JSON and the whole vector as raw float32.
// usage: logits_probe <model.pte> <ids.txt> <offset> <length> <out prefix>
//   -> <out prefix>.json {"n_tokens", "vocab", "next_id" (the id that follows the window, or -1), "top10"}
//      <out prefix>.f32  vocab float32 values
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
using executorch::extension::Module;
using executorch::extension::from_blob;
int main(int argc, char** argv) {
  if (argc != 6) { std::fprintf(stderr, "usage: %s model.pte ids.txt offset length out_prefix\n", argv[0]); return 2; }
  std::vector<int64_t> all; { std::ifstream f(argv[2]); int64_t v; while (f >> v) all.push_back(v); }
  const size_t off = std::strtoull(argv[3], nullptr, 10), len = std::strtoull(argv[4], nullptr, 10);
  if (len == 0 || off + len > all.size()) { std::fprintf(stderr, "window [%zu, %zu) outside %zu ids\n", off, off + len, all.size()); return 2; }
  std::vector<int64_t> ids(all.begin() + off, all.begin() + off + len);
  const long long next_id = off + len < all.size() ? (long long)all[off + len] : -1;
  Module m(argv[1], Module::LoadMode::MmapUseMlockIgnoreErrors);
  if (m.load_method("forward") != executorch::runtime::Error::Ok) { std::fprintf(stderr, "load failed\n"); return 1; }
  auto tok = from_blob(ids.data(), {1, (int)ids.size()}, executorch::aten::ScalarType::Long);
  int64_t pos0 = 0; auto pos = from_blob(&pos0, {1}, executorch::aten::ScalarType::Long);
  auto r = m.forward({tok, pos});
  if (!r.ok()) { std::fprintf(stderr, "forward failed %d\n", (int)r.error()); return 1; }
  auto t = r.get()[0].toTensor();
  const int64_t V = t.size(t.dim() - 1);
  if (t.scalar_type() != executorch::aten::ScalarType::Float) { std::fprintf(stderr, "unexpected dtype %d\n", (int)t.scalar_type()); return 1; }
  const float* lg = t.const_data_ptr<float>() + (t.numel() - V);
  std::vector<int> idx(V); std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
  const std::string pre(argv[5]);
  FILE* o = std::fopen((pre + ".json").c_str(), "w"); if (!o) { std::fprintf(stderr, "cannot write %s.json\n", argv[5]); return 1; }
  std::fprintf(o, "{\"n_tokens\": %zu, \"offset\": %zu, \"vocab\": %lld, \"next_id\": %lld, \"top10\": [", ids.size(), off, (long long)V, next_id);
  for (int i = 0; i < 10; i++) std::fprintf(o, "%s[%d, %.6f]", i ? ", " : "", idx[i], lg[idx[i]]);
  std::fprintf(o, "]}\n"); std::fclose(o);
  FILE* b = std::fopen((pre + ".f32").c_str(), "wb"); if (!b) return 1;
  std::fwrite(lg, sizeof(float), (size_t)V, b); std::fclose(b);
  std::printf("top1 %d %.4f n=%zu\n", idx[0], lg[idx[0]], ids.size());
  return 0;
}
