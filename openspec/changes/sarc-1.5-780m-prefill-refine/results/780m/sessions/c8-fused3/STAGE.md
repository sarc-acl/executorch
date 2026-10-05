session c8-fused3 staged 2026-10-05T04:10:25Z
parent = build/fused5 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3] commit 77cb45c06b99f4bc072277ea29d4a0fdce9827d4
cand   = build/fused5 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk] commit 77cb45c06b99f4bc072277ea29d4a0fdce9827d4
candidate 8: fused prefill SDPA kernel (sarc_dev_780m_sdpa_fused3, tile-packed K/V copies) through hooks/sdpa-fused-hook.patch in a scratch tree; both arms are the same binary (HEAD 77cb45c06 + working tree + both hook patches); the parent arm is candidate 7 (780m-refine3 + softmax r3)
e92a8bfe90439cda9916e242ba0a424b0caf06c99caff84fa6ae63b1c3b671a0  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c8-fused3/parent/llama_main
e92a8bfe90439cda9916e242ba0a424b0caf06c99caff84fa6ae63b1c3b671a0  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c8-fused3/cand/llama_main
a4f1c7c3acb9e1e895e07d3520f013dbcf18df4d40b31e039a184094429f1714  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c8-fused3/test_llama_microbench
