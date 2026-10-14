session final3 staged 2026-10-09T13:24:12Z
parent = build/7900xtx/parent env [ET_VK_SARC_UNVERIFIED=1] commit 90fe4d013dd39a6fcf7d68fd5f8038de5324bca3
cand   = build/7900xtx/c10 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5] commit 5950764fa336ee4aa6caed800bbf07bbcd6279be
driver = AMDVLK 2025.Q2.1 (LLPC), VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json (set by env.sh on the GPU host)
final stack (profile 7900xtx-refine5) on the build c10 = commit 5950764fa = the code of the committed head (git diff --name-only 5950764fa HEAD outside openspec is empty), no local patch, against the pristine parent 90fe4d013
ed0c2f21f5185b07b5e2c5d87b8c33ffa8a200103150a7b998f676390326e183  <artifacts>/stage/final3/parent/llama_main
e9d03689d091683021bc60b0246d1217c20d10b163083598a6b7fdcfd0c21af0  <artifacts>/stage/final3/cand/llama_main
5d7330cfe23e06396a5604e2cba0d9be5d624691f3205e6532098a1522b54c97  <artifacts>/stage/final3/parent/libllama_runner.so
8d1576d158d6dbe794c041577d4a8674559b9345a1761f2a0439ba5aad8ebc30  <artifacts>/stage/final3/cand/libllama_runner.so
c2296ddbcc02ccafde558486c7d41c0f6b70385bb2762382f207a4f56f937a64  <artifacts>/stage/final3/test_llama_microbench
