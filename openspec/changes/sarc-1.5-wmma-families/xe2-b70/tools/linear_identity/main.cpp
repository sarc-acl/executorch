#define main historical_microbench_main
#include <executorch/backends/vulkan/test/sarc_dev/test_llama_microbench.cpp>
#undef main

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  const std::string profile = argv[1];
  const std::string scheme = argv[2];
  const std::string op = scheme == "4w" ? "linear_q4gsw" : "linear_dq8ca_q4gsw";
  std::cout << "model,scheme,storage,M,N,K,parent_kernel,candidate_kernel,elements,bit_mismatches\n";
  size_t failures = 0;
  for (const auto& model : kLinearModels) {
    for (const auto& projection : model.ops) {
      for (auto storage : {utils::kBuffer, utils::kTexture3D}) {
        LinearConfig config{2048, projection.K, projection.N, 128, op, 1, model.model, "prefill", projection.op_label};
        TestCase inputs = make_deterministic_correctness_case(config, op, storage);
        for (auto& input : inputs.inputs()) {
          input.ensure_data_generated(7321);
        }
        if (scheme == "8da4w") {
          auto& zero_points = inputs.inputs()[2].get_int8_data();
          for (size_t i = 0; i < zero_points.size(); ++i) {
            zero_points[i] = int8_t(int(i % 7) * 9 - 27);
          }
        }
        TestCase parent = inputs;
        TestCase candidate = inputs;
        unsetenv("ET_VK_SARC_B70_WMMA_PROFILE");
        const auto parent_result = execute_test_case(parent, 1, 1, 1, true);
        setenv("ET_VK_SARC_B70_WMMA_PROFILE", profile.c_str(), 1);
        const auto candidate_result = execute_test_case(candidate, 1, 1, 1, true);
        const auto& a = parent.outputs()[0].get_half_data();
        const auto& b = candidate.outputs()[0].get_half_data();
        if (a.empty() || a.size() != b.size()) {
          return 1;
        }
        size_t mismatches = 0;
        for (size_t i = 0; i < a.size(); ++i) {
          mismatches += a[i] != b[i];
        }
        failures += mismatches != 0;
        std::cout << model.model << ',' << scheme << ','
                  << (storage == utils::kBuffer ? "buffer" : "texture3d")
                  << ",2048," << projection.N << ',' << projection.K << ','
                  << linear_kernel(parent_result) << ',' << linear_kernel(candidate_result)
                  << ',' << a.size() << ',' << mismatches << '\n';
      }
    }
  }
  std::cout << "LINEAR_IDENTITY_" << (failures ? "FAIL" : "PASS") << '\n';
  return failures != 0;
}
