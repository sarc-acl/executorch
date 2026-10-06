session s9-c4r staged 2026-10-06T05:22:32Z
parent = build/topic14 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf] commit aa66ea1e672e7df99a949ca26634f3cbe7edc0d1
cand   = build/topic14 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64] commit aa66ea1e672e7df99a949ca26634f3cbe7edc0d1
candidate 4 again on the corrected build: softmax orin_g64 (elected-lane store) on top of orin-refine5 + 4070ti_nzf (the parent arm)
source /mnt/linux-share/hmz-campaigns/jetson/.artifacts/orin-prefill-refine/build/topic14/source/executorch (tree aa66ea1e672e7df99a949ca26634f3cbe7edc0d1 -> /mnt/linux-share/hmz-campaigns/jetson/.artifacts/orin-prefill-refine/build/topic14/source/executorch (hard links to topic13, 179 changed paths written,; 30 submodules)
tree-sha256 topic14: not recomputed: hard links to topic13 (not...) + changed files verified by blob hash
source /mnt/linux-share/hmz-campaigns/jetson/.artifacts/orin-prefill-refine/build/topic14/source/executorch (tree aa66ea1e672e7df99a949ca26634f3cbe7edc0d1 -> /mnt/linux-share/hmz-campaigns/jetson/.artifacts/orin-prefill-refine/build/topic14/source/executorch (hard links to topic13, 179 changed paths written,; 30 submodules)
tree-sha256 topic14: not recomputed: hard links to topic13 (not...) + changed files verified by blob hash
c756572d2d1da7c21ee14274dc53726b2dbfa627b612b4233f4a8ce4043214cf  /home/doremy/hmz-sarc-orin/stage/s9-c4r/parent/COMMIT
edcbcafe5633644489567819077ec7d2bffafedfb581248dcde8dea524148444  /home/doremy/hmz-sarc-orin/stage/s9-c4r/parent/env
ab18f0a89b025cd802c35c36639af84326de1b8b1829c21efdd8f82cce01a9ff  /home/doremy/hmz-sarc-orin/stage/s9-c4r/parent/libllama_runner.so
71affa1baaf2dbd70ce2ada505e96d35a9654e94aa970a0f97f9179cfac07cdb  /home/doremy/hmz-sarc-orin/stage/s9-c4r/parent/llama_main
c756572d2d1da7c21ee14274dc53726b2dbfa627b612b4233f4a8ce4043214cf  /home/doremy/hmz-sarc-orin/stage/s9-c4r/cand/COMMIT
a357715cd2066b662ab1002055c3d79d3713f1a62966da7c35542d4e1cabeaf0  /home/doremy/hmz-sarc-orin/stage/s9-c4r/cand/env
ab18f0a89b025cd802c35c36639af84326de1b8b1829c21efdd8f82cce01a9ff  /home/doremy/hmz-sarc-orin/stage/s9-c4r/cand/libllama_runner.so
71affa1baaf2dbd70ce2ada505e96d35a9654e94aa970a0f97f9179cfac07cdb  /home/doremy/hmz-sarc-orin/stage/s9-c4r/cand/llama_main
71affa1baaf2dbd70ce2ada505e96d35a9654e94aa970a0f97f9179cfac07cdb  /home/doremy/hmz-sarc-orin/stage/s9-c4r/parent-traced/llama_main
71affa1baaf2dbd70ce2ada505e96d35a9654e94aa970a0f97f9179cfac07cdb  /home/doremy/hmz-sarc-orin/stage/s9-c4r/cand-traced/llama_main
71affa1baaf2dbd70ce2ada505e96d35a9654e94aa970a0f97f9179cfac07cdb  /home/doremy/hmz-sarc-orin/stage/s9-c4r/verify-bin/llama_main
195830599cc00a6b46f82313059a7098bc0502bff73e48833966c0c645ff5480  /home/doremy/hmz-sarc-orin/stage/s9-c4r/llama_main
233f6cd85bac2c6522fa4786a63bfd33b45ebb6b0a3128c418c2fdefcec96c11  /home/doremy/hmz-sarc-orin/stage/s9-c4r/test_llama_microbench
bfce65eb12a496801e29f6e8329773c20da1485d41f9fc504ce31549008b46e3  /home/doremy/hmz-sarc-orin/stage/s9-c4r/prompt_2048.txt
b5499448f07a40725ce9b96bb5094cb7bfa7ed749e8937213077b24b292445ba  /home/doremy/hmz-sarc-orin/stage/s9-c4r/prompt_check.txt
30ec73a22ec50d71c7e3f3255d9ae800f79b634d162839323585ca51bfc1c366  /home/doremy/hmz-sarc-orin/stage/s9-c4r/prompt_real_2048.txt
881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a  /home/doremy/hmz-sarc-orin/stage/s9-c4r/r1304.txt
