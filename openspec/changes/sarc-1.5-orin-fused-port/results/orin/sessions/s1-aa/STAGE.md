session s1-aa staged 2026-10-08T20:35:13Z
parent = build/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64] commit 8973ced76c6e346f34dc5b31b4755c423ceabc99
cand   = build/topic1 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64] commit 0f14f2a1a4e8dc5f8cb867860bc456fc913dd81d
baseline + A/A: the parent build against topic1 (hook + candidate 1 code), both with the parent environment
source /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/parent/source/executorch (tree 8973ced76c6e346f34dc5b31b4755c423ceabc99 -> /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/parent/source/executorch (30 submodules); 30 submodules)
tree-sha256 parent: 3bbbd4cbd0e32d97dac4231925baea43ff0bfc2e5e10be4d4f35bdd454d5c045
source /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/topic1/source/executorch (tree 0f14f2a1a4e8dc5f8cb867860bc456fc913dd81d -> /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/topic1/source/executorch (hard links to parent, 77 changed paths written, file list checke; 30 submodules)
tree-sha256 topic1: not recomputed: hard links to parent (3bbbd4cbd0e32d97...) + changed files verified by blob hash
8bddd0fa8a871957db4c40e1f68aca91283be27bcab0be353624e1a43aeccde8  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/parent/COMMIT
a357715cd2066b662ab1002055c3d79d3713f1a62966da7c35542d4e1cabeaf0  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/parent/env
ab18f0a89b025cd802c35c36639af84326de1b8b1829c21efdd8f82cce01a9ff  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/parent/libllama_runner.so
f0b8ff26bcdc742d69660a920d2d8cdc1f40a49f4c22d6ce3893d653e25c2fc3  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/parent/llama_main
b374c10965be3f6de34acd50bd99a1539b0b765cd7e7968505711805d77b836e  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/cand/COMMIT
a357715cd2066b662ab1002055c3d79d3713f1a62966da7c35542d4e1cabeaf0  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/cand/env
ab18f0a89b025cd802c35c36639af84326de1b8b1829c21efdd8f82cce01a9ff  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/cand/libllama_runner.so
16cf8606502e1922183a67258ad8519507a74ee2d7f37bca0dc91ed9e87f0565  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/cand/llama_main
f0b8ff26bcdc742d69660a920d2d8cdc1f40a49f4c22d6ce3893d653e25c2fc3  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/parent-traced/llama_main
16cf8606502e1922183a67258ad8519507a74ee2d7f37bca0dc91ed9e87f0565  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/cand-traced/llama_main
16cf8606502e1922183a67258ad8519507a74ee2d7f37bca0dc91ed9e87f0565  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/verify-bin/llama_main
195830599cc00a6b46f82313059a7098bc0502bff73e48833966c0c645ff5480  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/llama_main
d37968e68a6d0b9da08ea341ef470aba2e8671eccc6fdd0e1bd7557580c17059  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/test_llama_microbench
bfce65eb12a496801e29f6e8329773c20da1485d41f9fc504ce31549008b46e3  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/prompt_2048.txt
b5499448f07a40725ce9b96bb5094cb7bfa7ed749e8937213077b24b292445ba  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/prompt_check.txt
30ec73a22ec50d71c7e3f3255d9ae800f79b634d162839323585ca51bfc1c366  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/prompt_real_2048.txt
881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a  /home/doremy/hmz-sarc-orin-fused/stage/s1-aa/r1304.txt
