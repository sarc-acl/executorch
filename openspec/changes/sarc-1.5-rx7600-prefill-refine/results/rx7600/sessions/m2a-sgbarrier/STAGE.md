session m2a-sgbarrier staged 2026-10-08T04:36:13Z
parent = build/rx7600/c3 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2] commit 6ebf3948496b37ce15acd9f179435b3da6c6dd55
cand   = build/rx7600/c4 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2] commit 129cea7ac3ca1737a5e071d4673b96ac1ba1eb88
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
M2a: fused3sb (subgroupBarrier after every memoryBarrierShared) on candidate 3 (rx7600-refine2), build c4 against build c3
181e318c139116231eefd8d9ed689cc510b59b15272172cd9da56bb49bfa0f3f  <artifacts>/stage/m2a-sgbarrier/parent/llama_main
7cb695f20768a5942ab18727b9c3d0eb903957396c10bc65f87c5a2425cb9e02  <artifacts>/stage/m2a-sgbarrier/cand/llama_main
cefdc76a3bbc352c1d813b37daeb21521ea96b46b713966089890935b1f7a373  <artifacts>/stage/m2a-sgbarrier/parent/libllama_runner.so
118cb7ca09d63f0da1b6e957ea95fc8a46e788a05053ef18fbe35b92b00846a6  <artifacts>/stage/m2a-sgbarrier/cand/libllama_runner.so
740eefc99b0810d7a651ab35ef0bd192eb4feeeed8e46a83384bf3a858ba198d  <artifacts>/stage/m2a-sgbarrier/test_llama_microbench
