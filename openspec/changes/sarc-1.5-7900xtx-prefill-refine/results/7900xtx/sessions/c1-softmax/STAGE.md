session c1-softmax staged 2026-10-08T18:18:36Z
parent = build/7900xtx/parent env [ET_VK_SARC_UNVERIFIED=1] commit 90fe4d013dd39a6fcf7d68fd5f8038de5324bca3
cand   = build/7900xtx/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7] commit 90fe4d013dd39a6fcf7d68fd5f8038de5324bca3
driver = AMDVLK 2025.Q2.1 (LLPC), VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json (set by env.sh on the GPU host)
candidate 1: softmax r3 of the 780M (ET_VK_SARC_780M_PROFILE=c7) on the parent binary
ed0c2f21f5185b07b5e2c5d87b8c33ffa8a200103150a7b998f676390326e183  <artifacts>/stage/c1-softmax/parent/llama_main
ed0c2f21f5185b07b5e2c5d87b8c33ffa8a200103150a7b998f676390326e183  <artifacts>/stage/c1-softmax/cand/llama_main
5d7330cfe23e06396a5604e2cba0d9be5d624691f3205e6532098a1522b54c97  <artifacts>/stage/c1-softmax/parent/libllama_runner.so
5d7330cfe23e06396a5604e2cba0d9be5d624691f3205e6532098a1522b54c97  <artifacts>/stage/c1-softmax/cand/libllama_runner.so
6ff910b6a02f9f5e1210e8e1177a116fc5c99653d0e90a0e962e48771a801334  <artifacts>/stage/c1-softmax/test_llama_microbench
