session r2-final-r1 staged 2026-10-09T10:16:06Z
parent = build/rx7600/final env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2] commit 18cc0d53a91e42f1100cc08c7f2c8327c1c77770
cand   = build/rx7600/f2 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine5] commit 73648f5bd6f6005060e7dd7e7b6ecbb49ad7c4e4
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
round 2 final stack (build f2) against round 1's final stack (build final, rx7600-refine2)
ca2d39be9c70a2ccbcb5267ecadffffb34f43f4730de3c2f14ae5d7b829a47b6  <artifacts>/stage/r2-final-r1/parent/llama_main
b062ca10c4a96dd9f6ceba615222fe0032ecf7e4f76929bfb82cb61da8d019f0  <artifacts>/stage/r2-final-r1/cand/llama_main
4cf4dab6fcd3817364c33f26bf31ce8d48d2d1e83e148ae1848c4d3d45cbb0c5  <artifacts>/stage/r2-final-r1/parent/libllama_runner.so
36ab49fa06ca907290f87f787f3d60efde0de8f8daeb5c9748f15188d12ed260  <artifacts>/stage/r2-final-r1/cand/libllama_runner.so
c667bc7235ab30c1f563a9eb7850608eb1508d2d0641452b1aa0fe2d52c1bbcb  <artifacts>/stage/r2-final-r1/test_llama_microbench
