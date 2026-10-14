session s2-c1 staged 2026-10-04T21:20:15Z
parent = build/parent env [] commit 6a7cc8cc643a8c973d42ba6946839723df18078b
cand   = build/topic4 env [ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1] commit d3df2bb69d31cfee2fcf8522f71dfb7fe4324abf
candidate 1: SDPA prefill kernels (profile 4070ti-refine1) against the pristine parent
source /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/src/parent/executorch (git archive + pinned submodules, 30 submodules)
tree-sha256 parent: a822ac2a62e7ffe79242b6d5d00c0dc10a452c344c94d7629e18fe4bbd5d8f89
source /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/src/topic4/executorch (git archive + pinned submodules, 30 submodules)
local-patch topic4: local-hook-nvidia-sdpa.patch sha256 7c5b43f75695c05a86d342e8c466bff72b71432e1315aa19302a2f5c4fd1c359 (NOT committed; files: backends/vulkan/runtime/graph/ops/impl/sarc/table_nvidia.cpp )
tree-sha256 topic4: 3819782cb60abdb0da97bf814d80ea25cf2f1ab1dab49252017da5c2b0f62d25
4291fddf065fd4c0a49579b4263173797dfbfa689514e8862067af952da3f73e  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/parent/COMMIT
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/parent/env
83741af364c29958dc1aeff2e8012cf84611952260d882b05a2bb75b55da567c  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/parent/libllama_runner.so
82ac4987c4e0789fd46859c5fa053b886df67c77880095388d635f06dbdcd08a  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/parent/llama_main
66d6acf66dc8e68c39831dc224b9114faecba62adaa87d572d4d80ba97e896d9  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/cand/COMMIT
3769843a5c375e4b67de68e7896ea79dfb86ab5d02c63eb519cd2cdfd525529f  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/cand/env
0d3ce79708b0c9c17ced86c8a8b64845ab6ff362cf9b5e09b318436555c69bd6  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/cand/libllama_runner.so
ddcb612ae7b13cce7445062ff4bb9d4bb23950623c3503ed2320007c23750f5f  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/cand/llama_main
513f0ebf4a9149edd909b57c7fb00a9663d35633962206510b820c580ed85d78  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/parent-traced/llama_main
a18c34296e4b9e87a7bd4fdeadcb18edec8eefbc84e3b1f8dc013a616a917b7a  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/cand-traced/llama_main
ddcb612ae7b13cce7445062ff4bb9d4bb23950623c3503ed2320007c23750f5f  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/verify-bin/llama_main
195830599cc00a6b46f82313059a7098bc0502bff73e48833966c0c645ff5480  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/llama_main
6ae910f889d8a458dff750f3196e3df19355e0cf9ddc3dcef5162a681f2035e5  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/test_llama_microbench
bfce65eb12a496801e29f6e8329773c20da1485d41f9fc504ce31549008b46e3  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/prompt_2048.txt
b5499448f07a40725ce9b96bb5094cb7bfa7ed749e8937213077b24b292445ba  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/prompt_check.txt
30ec73a22ec50d71c7e3f3255d9ae800f79b634d162839323585ca51bfc1c366  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/prompt_real_2048.txt
881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a  /home/doremy/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/stage/s2-c1/r1304.txt
