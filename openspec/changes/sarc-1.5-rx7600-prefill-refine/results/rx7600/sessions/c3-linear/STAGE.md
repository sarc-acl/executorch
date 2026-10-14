session c3-linear staged 2026-10-08T01:52:16Z
parent = build/rx7600/parent env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko] commit f5f1bf10c510da347a556f716c2c9025a85d6228
cand   = build/rx7600/c3 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2] commit 6ebf3948496b37ce15acd9f179435b3da6c6dd55
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
candidate 3: linear kernel per layer shape (profile rx7600-refine2, build c3 = commit 6ebf39484) against candidate 2 (parent binary, env switches)
cfb5fb8ca692e4cef9d414826eab73bf52aa24d1ad4249b3a13465c59bc87c9d  <artifacts>/stage/c3-linear/parent/llama_main
181e318c139116231eefd8d9ed689cc510b59b15272172cd9da56bb49bfa0f3f  <artifacts>/stage/c3-linear/cand/llama_main
9244aefaebb940f48df675576b281bdfe4cfbea9ee63476d09130d1c7c4b0ab4  <artifacts>/stage/c3-linear/parent/libllama_runner.so
cefdc76a3bbc352c1d813b37daeb21521ea96b46b713966089890935b1f7a373  <artifacts>/stage/c3-linear/cand/libllama_runner.so
ed71415a5199eb48e34e517b4656887edced1872c06890e033b1efe902baa9bb  <artifacts>/stage/c3-linear/test_llama_microbench
