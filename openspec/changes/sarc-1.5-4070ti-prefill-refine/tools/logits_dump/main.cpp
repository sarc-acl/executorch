// logits_dump: the full last-position logits of a Llama .pte (Vulkan delegate) for many prompts, one process.
// Build pattern and forward call are those of sarc-1.5-e2e-benchmark/kit/logits_probe; this tool loads the model
// once, runs one prefill at position 0 per line of <ids file> (space-separated token ids, one prompt per line)
// and appends the vocabulary-sized float32 logits of the last position to <out.bin>.
// usage: logits_dump <model.pte> <ids file> <out.bin>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using executorch::extension::Module;
using executorch::extension::from_blob;
int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: %s model.pte ids.txt out.bin\n", argv[0]); return 2; }
  Module m(argv[1], Module::LoadMode::MmapUseMlockIgnoreErrors);
  if (m.load_method("forward") != executorch::runtime::Error::Ok) { std::fprintf(stderr, "load failed\n"); return 1; }
  std::ifstream f(argv[2]); std::string line; FILE* o = std::fopen(argv[3], "wb");
  if (!f || o == nullptr) { std::fprintf(stderr, "cannot open ids or output\n"); return 1; }
  int n = 0;
  while (std::getline(f, line)) {
    std::vector<int64_t> ids; { std::istringstream s(line); int64_t v; while (s >> v) ids.push_back(v); }
    if (ids.empty()) continue;
    auto tok = from_blob(ids.data(), {1, (int)ids.size()}, executorch::aten::ScalarType::Long);
    int64_t pos0 = 0; auto pos = from_blob(&pos0, {1}, executorch::aten::ScalarType::Long);
    auto r = m.forward({tok, pos});
    if (!r.ok()) { std::fprintf(stderr, "forward failed %d on prompt %d\n", (int)r.error(), n); return 1; }
    auto t = r.get()[0].toTensor();
    const int64_t V = t.size(t.dim() - 1);
    if (t.scalar_type() != executorch::aten::ScalarType::Float) { std::fprintf(stderr, "unexpected dtype\n"); return 1; }
    const float* p = t.const_data_ptr<float>() + (t.numel() - V);
    std::fwrite(p, sizeof(float), V, o);
    int best = 0; for (int64_t i = 1; i < V; ++i) if (p[i] > p[best]) best = (int)i;
    std::printf("prompt %d tokens %zu vocab %lld top1 %d %.6f\n", n, ids.size(), (long long)V, best, p[best]);
    ++n;
  }
  std::fclose(o); std::printf("prompts %d\n", n);
  return 0;
}
