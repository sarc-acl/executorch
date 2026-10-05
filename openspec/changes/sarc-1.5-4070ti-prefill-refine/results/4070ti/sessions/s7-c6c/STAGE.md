session s7-c6c staged 2026-10-05T08:34:44Z
parent = build/topic11 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf ET_VK_SARC_DEV_PROFILE=4070ti-refine1] commit 42e001462a2ee5c9142b2689f09db182894381a9
cand   = build/topic11 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf ET_VK_SARC_DEV_PROFILE=4070ti-refine5] commit 42e001462a2ee5c9142b2689f09db182894381a9
candidate 6: candidate 4 + 8da4w half-texel staging + 4w wide tile with the texture3d drain in Ash (profile 4070ti-refine5), attempt s7-c6c, against candidate 4 (same build, profile 4070ti-refine1)
source /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/src/topic11/executorch (git archive + pinned submodules, 30 submodules)
local-patch topic11: local-hook-nvidia-sdpa-softmax.patch sha256 20bd904b5b51318095974d68e9a0835b5812d9d1372852d42873f91f337a6273 (NOT committed; files: backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp backends/vulkan/runtime/graph/ops/impl/sarc/table_nvidia.cpp )
tree-sha256 topic11: 2d6152261bfa880d03e7208473ff71a64c8d9ea402aa913dde4ce2c07ba26a8a
source /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/src/topic11/executorch (git archive + pinned submodules, 30 submodules)
local-patch topic11: local-hook-nvidia-sdpa-softmax.patch sha256 20bd904b5b51318095974d68e9a0835b5812d9d1372852d42873f91f337a6273 (NOT committed; files: backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp backends/vulkan/runtime/graph/ops/impl/sarc/table_nvidia.cpp )
tree-sha256 topic11: 2d6152261bfa880d03e7208473ff71a64c8d9ea402aa913dde4ce2c07ba26a8a
190d53bcbbdb42ee976e440cd4d53da4ba9cb02f7997fc37f1d860b7af429bca  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/parent/COMMIT
9f60c46453e5806ea42e63b4f4a70f4aa2ecaa30413cb9af0b5163d0bc52d36c  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/parent/env
68afd65b81b2ffa333526313eac39df7d5ccb5efa4827ad89d1c678dab825277  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/parent/libllama_runner.so
42a7903dbab828b540851ea4b1f53feac5f7ae842a491d1ad445a558f9867d88  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/parent/llama_main
190d53bcbbdb42ee976e440cd4d53da4ba9cb02f7997fc37f1d860b7af429bca  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/cand/COMMIT
0382dacc63fddb872f1b2a01b8fa0ed9c8e3805b3c731c34a9a5092bf46988d8  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/cand/env
68afd65b81b2ffa333526313eac39df7d5ccb5efa4827ad89d1c678dab825277  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/cand/libllama_runner.so
42a7903dbab828b540851ea4b1f53feac5f7ae842a491d1ad445a558f9867d88  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/cand/llama_main
72cd5e6a407efcd1c110ad4051cd40148885650d5f4bc2b7cdc1467fab1280cb  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/parent-traced/llama_main
72cd5e6a407efcd1c110ad4051cd40148885650d5f4bc2b7cdc1467fab1280cb  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/cand-traced/llama_main
42a7903dbab828b540851ea4b1f53feac5f7ae842a491d1ad445a558f9867d88  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/verify-bin/llama_main
195830599cc00a6b46f82313059a7098bc0502bff73e48833966c0c645ff5480  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/llama_main
79b12fcfe411536a14ffe58e75d65df772ce59d6da7552a0b473ad16cc55a8f2  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/test_llama_microbench
bfce65eb12a496801e29f6e8329773c20da1485d41f9fc504ce31549008b46e3  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/prompt_2048.txt
b5499448f07a40725ce9b96bb5094cb7bfa7ed749e8937213077b24b292445ba  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/prompt_check.txt
30ec73a22ec50d71c7e3f3255d9ae800f79b634d162839323585ca51bfc1c366  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/prompt_real_2048.txt
881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s7-c6c/r1304.txt
