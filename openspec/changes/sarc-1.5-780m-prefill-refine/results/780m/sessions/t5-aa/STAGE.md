session t5-aa staged 2026-10-05T17:28:37Z
parent = build/fused9 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_780M_PROFILE=refine10] commit 7301979208b1fe2723ee61fdb02c4cec6b1e1344
cand   = build/fused9 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_780M_PROFILE=refine10] commit 7301979208b1fe2723ee61fdb02c4cec6b1e1344
post-reboot A/A, 3 repeats: build fused9 with candidate 10's environment on both arms
2762a049563b9cab03cc31450f2600fb78745033ec83f4e163c187e6c3cfb8a8  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-05/stage/t5-aa/parent/llama_main
2762a049563b9cab03cc31450f2600fb78745033ec83f4e163c187e6c3cfb8a8  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-05/stage/t5-aa/cand/llama_main
3df42eb9fb96a8ace7253b177d4620e6a07e1ade232f091edfe0cbbab31d4973  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-05/stage/t5-aa/test_llama_microbench
