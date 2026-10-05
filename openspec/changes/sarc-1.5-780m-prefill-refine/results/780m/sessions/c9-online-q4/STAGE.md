session c9-online-q4 staged 2026-10-05T08:58:56Z
parent = build/fused7 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk] commit 326ea868dd83413e8b578f7dec176df15f765b9d
cand   = build/fused7 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_780M_PROFILE=refine9] commit 326ea868dd83413e8b578f7dec176df15f765b9d
candidate 9: candidate 8 + the one-pass form of the fused SDPA kernel (fused3 ...rko) + the 4w kernel per shape of the round-2 search (ET_VK_SARC_780M_PROFILE=refine9); both arms are the same binary (HEAD 326ea868d + working tree + both hook patches); the parent arm is candidate 8
e96280beca3503613f8185b083053a29ffb26dfc472a66f5931e9df1d70b9b4e  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c9-online-q4/parent/llama_main
e96280beca3503613f8185b083053a29ffb26dfc472a66f5931e9df1d70b9b4e  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c9-online-q4/cand/llama_main
3063934fc7345ed35353d615ef494fc600624e4ff8604f1eafe560eb337345af  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c9-online-q4/test_llama_microbench
