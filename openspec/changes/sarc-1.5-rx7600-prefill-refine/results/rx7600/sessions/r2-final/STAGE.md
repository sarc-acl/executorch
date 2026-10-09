session r2-final staged 2026-10-09T06:17:13Z
parent = build/rx7600/parent env [ET_VK_SARC_UNVERIFIED=1] commit f5f1bf10c510da347a556f716c2c9025a85d6228
cand   = build/rx7600/f2 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine5] commit 73648f5bd6f6005060e7dd7e7b6ecbb49ad7c4e4
driver = <mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json (Mesa 26.2.3 31e9a6b2e9)
round 2 final stack (rx7600-refine5 + fused3sb + softmax r3, build f2 = commit 73648f5bd6f6005060e7dd7e7b6ecbb49ad7c4e4) against the pristine parent f5f1bf10c
cfb5fb8ca692e4cef9d414826eab73bf52aa24d1ad4249b3a13465c59bc87c9d  <artifacts>/stage/r2-final/parent/llama_main
b062ca10c4a96dd9f6ceba615222fe0032ecf7e4f76929bfb82cb61da8d019f0  <artifacts>/stage/r2-final/cand/llama_main
9244aefaebb940f48df675576b281bdfe4cfbea9ee63476d09130d1c7c4b0ab4  <artifacts>/stage/r2-final/parent/libllama_runner.so
36ab49fa06ca907290f87f787f3d60efde0de8f8daeb5c9748f15188d12ed260  <artifacts>/stage/r2-final/cand/libllama_runner.so
c667bc7235ab30c1f563a9eb7850608eb1508d2d0641452b1aa0fe2d52c1bbcb  <artifacts>/stage/r2-final/test_llama_microbench
