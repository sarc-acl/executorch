session c10-q4-refine10 staged 2026-10-05T15:17:14Z
parent = build/fused9 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_780M_PROFILE=refine9] commit 7301979208b1fe2723ee61fdb02c4cec6b1e1344
cand   = build/fused9 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_780M_PROFILE=refine10] commit 7301979208b1fe2723ee61fdb02c4cec6b1e1344
candidate 10: candidate 9 with the 4w kernel per shape after two more refinement rounds (ET_VK_SARC_780M_PROFILE=refine10 instead of refine9); both arms are the same binary (HEAD 730197920 + both hook patches); the parent arm is candidate 9
2762a049563b9cab03cc31450f2600fb78745033ec83f4e163c187e6c3cfb8a8  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c10-q4-refine10/parent/llama_main
2762a049563b9cab03cc31450f2600fb78745033ec83f4e163c187e6c3cfb8a8  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c10-q4-refine10/cand/llama_main
3df42eb9fb96a8ace7253b177d4620e6a07e1ade232f091edfe0cbbab31d4973  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c10-q4-refine10/test_llama_microbench
