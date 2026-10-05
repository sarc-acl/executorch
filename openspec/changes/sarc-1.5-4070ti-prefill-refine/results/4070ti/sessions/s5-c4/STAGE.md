session s5-c4 staged 2026-10-05T00:46:06Z
parent = build/parent env [] commit 6a7cc8cc643a8c973d42ba6946839723df18078b
cand   = build/topic10 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf] commit 267edc6a5e51f03a1485bd60b105f02b9bc9644f
candidate 4: SDPA prefill kernels (profile 4070ti-refine1) with the fp32, no-zero-tail softmax (4070ti_nzf; hooks 1 and 2 by local patch) against the pristine parent
source /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/src/parent/executorch (git archive + pinned submodules, 30 submodules)
tree-sha256 parent: a822ac2a62e7ffe79242b6d5d00c0dc10a452c344c94d7629e18fe4bbd5d8f89
source /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/src/topic10/executorch (git archive + pinned submodules, 30 submodules)
local-patch topic10: local-hook-nvidia-sdpa-softmax.patch sha256 20bd904b5b51318095974d68e9a0835b5812d9d1372852d42873f91f337a6273 (NOT committed; files: backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp backends/vulkan/runtime/graph/ops/impl/sarc/table_nvidia.cpp )
tree-sha256 topic10: 7a428196dd409ef7f2c84b9267dc2cceea8a212384fb6dff1b4a0704c1533e36
4291fddf065fd4c0a49579b4263173797dfbfa689514e8862067af952da3f73e  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/parent/COMMIT
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/parent/env
83741af364c29958dc1aeff2e8012cf84611952260d882b05a2bb75b55da567c  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/parent/libllama_runner.so
82ac4987c4e0789fd46859c5fa053b886df67c77880095388d635f06dbdcd08a  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/parent/llama_main
d7267bb9640c254fa5d4930d10c3b1fc2c1d60071623523757f92b36cd2273c7  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/cand/COMMIT
d9b81aa03ec20b6f2c3cb0897d3104e48dafe9e123df9c3aabd74f6171813e3f  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/cand/env
c350207c227eec2c24b72c980ca3a7118db9a42b3d85fafdee5663d8695584cf  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/cand/libllama_runner.so
581f5adcdd7563bdc19486a0aa45e69ddfe1d1ab95eb149346968ba8234fb936  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/cand/llama_main
513f0ebf4a9149edd909b57c7fb00a9663d35633962206510b820c580ed85d78  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/parent-traced/llama_main
4ed0e7a0103cba3e2954f37e0511e95d30a27b411a484f9babbce8b429d4e0cb  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/cand-traced/llama_main
581f5adcdd7563bdc19486a0aa45e69ddfe1d1ab95eb149346968ba8234fb936  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/verify-bin/llama_main
195830599cc00a6b46f82313059a7098bc0502bff73e48833966c0c645ff5480  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/llama_main
dce112e3708f8cea0b4ccf359e68cf3736ba3ac3e3ec783b929c4d1d4254b17c  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/test_llama_microbench
bfce65eb12a496801e29f6e8329773c20da1485d41f9fc504ce31549008b46e3  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/prompt_2048.txt
b5499448f07a40725ce9b96bb5094cb7bfa7ed749e8937213077b24b292445ba  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/prompt_check.txt
30ec73a22ec50d71c7e3f3255d9ae800f79b634d162839323585ca51bfc1c366  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/prompt_real_2048.txt
881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s5-c4/r1304.txt
