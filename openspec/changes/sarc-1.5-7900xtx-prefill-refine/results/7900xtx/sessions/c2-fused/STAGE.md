session c2-fused staged 2026-10-08T19:56:30Z
parent = build/7900xtx/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7] commit 90fe4d013dd39a6fcf7d68fd5f8038de5324bca3
cand   = build/7900xtx/c2 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rk,fused3sb_d128_t16x64g11s32rk] commit 86a7f96c80243236dd7dbdf5f79205b96af6535f
driver = AMDVLK 2025.Q2.1 (LLPC), VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json (set by env.sh on the GPU host)
candidate 2: fused attention kernel fused3sb (M2a), rk variants picked by the fused-variant screen (stage c1-softmax/fused-screen.csv); parent arm = candidate 1 (parent binary, c7)
ed0c2f21f5185b07b5e2c5d87b8c33ffa8a200103150a7b998f676390326e183  <artifacts>/stage/c2-fused/parent/llama_main
29cf9d912a43ec807a0debf9877d4dc593e9251b17e087424b25b53a0420daaf  <artifacts>/stage/c2-fused/cand/llama_main
5d7330cfe23e06396a5604e2cba0d9be5d624691f3205e6532098a1522b54c97  <artifacts>/stage/c2-fused/parent/libllama_runner.so
d50106eb8784e2df0933bfd27fba63f011824f95b7f7c0cb1b0a242f20b8bc24  <artifacts>/stage/c2-fused/cand/libllama_runner.so
c7921bff7c1ed60ab63f786406e9cdd54edd66ec446267c2e735e06072a16566  <artifacts>/stage/c2-fused/test_llama_microbench
