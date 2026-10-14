session c2-fused staged 2026-10-06T13:22:47Z
parent = build/rx7600/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7] commit f5f1bf10c510da347a556f716c2c9025a85d6228
cand   = build/rx7600/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko] commit f5f1bf10c510da347a556f716c2c9025a85d6228
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
candidate 2: fused attention kernel (780M fused3, fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko) on top of candidate 1, parent binary
cfb5fb8ca692e4cef9d414826eab73bf52aa24d1ad4249b3a13465c59bc87c9d  <artifacts>/stage/c2-fused/parent/llama_main
cfb5fb8ca692e4cef9d414826eab73bf52aa24d1ad4249b3a13465c59bc87c9d  <artifacts>/stage/c2-fused/cand/llama_main
9244aefaebb940f48df675576b281bdfe4cfbea9ee63476d09130d1c7c4b0ab4  <artifacts>/stage/c2-fused/parent/libllama_runner.so
9244aefaebb940f48df675576b281bdfe4cfbea9ee63476d09130d1c7c4b0ab4  <artifacts>/stage/c2-fused/cand/libllama_runner.so
21cfb63da92705314aceaa0fbb61e519d1e7be22e8a8aa19f00c7619e9b79ee7  <artifacts>/stage/c2-fused/test_llama_microbench
