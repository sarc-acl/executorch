session r2-c7-q4 staged 2026-10-09T05:03:47Z
parent = build/rx7600/c6 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine4] commit 36c7d1cc0e48794efd4805629148492aa077feec
cand   = build/rx7600/c7 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine5] commit d6d67ba781330522057fbded3ff7eff594506025
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
round 2 candidate 2: 4w 256x128 tile with 72-byte A and 88-byte B staging rows (rx7600-refine5, build c7 = commit d6d67ba78) against candidate 1 (build c6, rx7600-refine4)
f33384887d6153dd2bc0456531ae4a730498019bb77063ddcefe5859fd65070a  <artifacts>/stage/r2-c7-q4/parent/llama_main
b1f5b84c472cc1bd0ad0b75c89ca80745a4a63ba9ef49c9c3422de9eaabd65cf  <artifacts>/stage/r2-c7-q4/cand/llama_main
2c5b83137ead1613dfeb1ced953c0f5e5be8345e4148f5c433e84234ad7e4be1  <artifacts>/stage/r2-c7-q4/parent/libllama_runner.so
8dac276ef47cb514071405e9c4003a6090a549e9fe550131a24674346b4725c0  <artifacts>/stage/r2-c7-q4/cand/libllama_runner.so
6356d62c6bfc5114111bd8a0aab174da1be7776fab9091c7137013279c10123d  <artifacts>/stage/r2-c7-q4/test_llama_microbench
