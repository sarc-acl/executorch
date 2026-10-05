session c7-softmax-r3 staged 2026-10-04T21:17:10Z
parent = build/hook3 env [ET_VK_SARC_DEV_PROFILE=780m-refine3] commit e12301fd5f4a62ded78a6ca9e6084ec06aa4b311
cand   = build/hook3 env [ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3] commit e12301fd5f4a62ded78a6ca9e6084ec06aa4b311
candidate 7: softmax variant r3 (single load, zero fill bounded to what the SARC attn*V kernels read) through hooks/softmax-name-hook.patch in a scratch tree; both arms are the same binary (HEAD e12301fd5 + working-tree softmax r3 and test changes + the hook patch), the parent arm without the softmax variable
51a09b7e991346b7059b9f1f4323bd1285d60e15b47968fbeb12e64a99d01ae1  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c7-softmax-r3/parent/llama_main
51a09b7e991346b7059b9f1f4323bd1285d60e15b47968fbeb12e64a99d01ae1  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c7-softmax-r3/cand/llama_main
a08f05713c3ec2a2e8ded9565aa9300d3dde14b117c627bb8d7ee0a36cf43538  /home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/stage/c7-softmax-r3/test_llama_microbench
