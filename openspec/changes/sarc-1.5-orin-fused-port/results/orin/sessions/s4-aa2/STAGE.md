session s4-aa2 staged 2026-10-09T05:06:40Z
parent = build/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64] commit 8973ced76c6e346f34dc5b31b4755c423ceabc99
cand   = build/topic4 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64] commit 0bed380905e134c73b56838e4e920e953ee1234a
A/A re-check with the thermal throttle record: the parent build against topic4, both with the parent environment
source /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/parent/source/executorch (tree 8973ced76c6e346f34dc5b31b4755c423ceabc99 -> /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/parent/source/executorch (30 submodules); 30 submodules)
tree-sha256 parent: 3bbbd4cbd0e32d97dac4231925baea43ff0bfc2e5e10be4d4f35bdd454d5c045
source /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/topic4/source/executorch (tree 0bed380905e134c73b56838e4e920e953ee1234a -> /mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/build/topic4/source/executorch (hard links to topic3, 7 changed paths written, file list checked; 30 submodules)
tree-sha256 topic4: not recomputed: hard links to topic3 (not...) + changed files verified by blob hash
8bddd0fa8a871957db4c40e1f68aca91283be27bcab0be353624e1a43aeccde8  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/parent/COMMIT
a357715cd2066b662ab1002055c3d79d3713f1a62966da7c35542d4e1cabeaf0  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/parent/env
ab18f0a89b025cd802c35c36639af84326de1b8b1829c21efdd8f82cce01a9ff  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/parent/libllama_runner.so
f0b8ff26bcdc742d69660a920d2d8cdc1f40a49f4c22d6ce3893d653e25c2fc3  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/parent/llama_main
15c4afcd4d75a3ad49912e1115bef825928742dd23b6f85f35943b2777fcf2e8  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/cand/COMMIT
a357715cd2066b662ab1002055c3d79d3713f1a62966da7c35542d4e1cabeaf0  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/cand/env
ab18f0a89b025cd802c35c36639af84326de1b8b1829c21efdd8f82cce01a9ff  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/cand/libllama_runner.so
d33b5330ec549c872c937ba5513529a937c1d9057670a4c4c6fae8faee17e357  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/cand/llama_main
f0b8ff26bcdc742d69660a920d2d8cdc1f40a49f4c22d6ce3893d653e25c2fc3  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/parent-traced/llama_main
d33b5330ec549c872c937ba5513529a937c1d9057670a4c4c6fae8faee17e357  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/cand-traced/llama_main
d33b5330ec549c872c937ba5513529a937c1d9057670a4c4c6fae8faee17e357  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/verify-bin/llama_main
195830599cc00a6b46f82313059a7098bc0502bff73e48833966c0c645ff5480  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/llama_main
72535d0f41674fdadb5ba54b66d04df6e796a19432f2832402e53218757e7793  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/test_llama_microbench
bfce65eb12a496801e29f6e8329773c20da1485d41f9fc504ce31549008b46e3  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/prompt_2048.txt
b5499448f07a40725ce9b96bb5094cb7bfa7ed749e8937213077b24b292445ba  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/prompt_check.txt
30ec73a22ec50d71c7e3f3255d9ae800f79b634d162839323585ca51bfc1c366  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/prompt_real_2048.txt
881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a  /home/doremy/hmz-sarc-orin-fused/stage/s4-aa2/r1304.txt
