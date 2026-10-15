#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>

#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  executorch::runtime::runtime_init();
  using executorch::extension::module::Module;
  Module module(argv[1], Module::LoadMode::Mmap);
  auto names = module.method_names();
  if (!names.ok()) {
    return 3;
  }
  const char* methods[] = {
      "get_max_seq_len", "get_max_context_len", "get_n_layers", "get_vocab_size",
      "use_kv_cache", "use_sdpa_with_kv_cache", "enable_dynamic_shape",
      "get_bos_id", "get_eos_ids"};
  std::ostringstream json;
  json << "{\"constant_methods\":{";
  bool first = true;
  for (const auto* name : methods) {
    auto meta = module.program()->method_meta(name);
    if (!meta.ok() || meta->num_inputs() != 0 || meta->num_backends() != 0 ||
        meta->num_instructions() != 0) {
      std::cerr << "Not a CPU-only constant method: " << name << '\n';
      return 4;
    }
    auto result = module.execute(name);
    if (!result.ok() || result->empty()) {
      std::cerr << "Constant method failed: " << name << '\n';
      return 5;
    }
    json << (first ? "" : ",") << '"' << name << "\":[";
    first = false;
    for (size_t i = 0; i < result->size(); ++i) {
      json << (i ? "," : "");
      const auto& value = (*result)[i];
      if (value.isInt()) {
        json << value.toInt();
      } else if (value.isBool()) {
        json << (value.toBool() ? "true" : "false");
      } else {
        std::cerr << "Unexpected constant type: " << name << '\n';
        return 6;
      }
    }
    json << ']';
  }
  std::cout << json.str() << "}}\n";
  return 0;
}
