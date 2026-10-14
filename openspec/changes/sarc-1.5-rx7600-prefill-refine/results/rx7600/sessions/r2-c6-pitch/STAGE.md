session r2-c6-pitch staged 2026-10-09T03:57:28Z
parent = build/rx7600/final env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2] commit 18cc0d53a91e42f1100cc08c7f2c8327c1c77770
cand   = build/rx7600/c6 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine4] commit 36c7d1cc0e48794efd4805629148492aa077feec
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
round 2 candidate 1: 8da4w A staging row pitch 24 bytes (rx7600-refine4, build c6 = commit 36c7d1cc0) against round 1 final stack (build final, rx7600-refine2)
ca2d39be9c70a2ccbcb5267ecadffffb34f43f4730de3c2f14ae5d7b829a47b6  <artifacts>/stage/r2-c6-pitch/parent/llama_main
f33384887d6153dd2bc0456531ae4a730498019bb77063ddcefe5859fd65070a  <artifacts>/stage/r2-c6-pitch/cand/llama_main
4cf4dab6fcd3817364c33f26bf31ce8d48d2d1e83e148ae1848c4d3d45cbb0c5  <artifacts>/stage/r2-c6-pitch/parent/libllama_runner.so
2c5b83137ead1613dfeb1ced953c0f5e5be8345e4148f5c433e84234ad7e4be1  <artifacts>/stage/r2-c6-pitch/cand/libllama_runner.so
eabf72e648f165e619d7e54b9fc1bf6961e8f82428e561a429dd9617634d8a9c  <artifacts>/stage/r2-c6-pitch/test_llama_microbench
