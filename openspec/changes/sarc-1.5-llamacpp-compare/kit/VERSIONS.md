# Versions

| what | version |
|---|---|
| llama.cpp | tag `b11430`, commit `8345f333951c661d166b00e6f9362e553768f292` (2026-10-05), pinned on 2026-10-05 for every device |
| llama.cpp fallback | tag `b10229`, commit `c745be2a2c5aefbf9f3ced0440804373731890b6` (the August build), only if the pinned commit fails on a device |
| ExecuTorch arms | recorded per device in `results/<device>/ARMS.md` |
| ROCm (llama.cpp HIP backend, RX 7900 XTX) | **7.2.1** (`rocm-core` 7.2.1.70201, HIP runtime 7.2.53211, AMD clang 22.0.0 `roc-7.2.1`, `hipblas` 3.2.0.70201, `rocblas` 5.2.0.70201), kernel driver `amdgpu` 6.16.13 of the host (unchanged). Built in the public container image `rocm/dev-ubuntu-24.04:7.2.1` (digest `sha256:fe29799baf7b8db986a707110c15579dc0fb6e39d48973fbb82dfc33a0c7e179`) with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`, run on the host with the container's runtime libraries beside the binaries; the host got no install. Details: `results/7900xtx/hip/ARMS.md` |

Build options and tool versions per backend and host are written by `build-llamacpp.sh` beside the binaries
and copied into `results/<device>/` with the first session of that device.

## Prompt token count under the pinned llama.cpp (measured 2026-10-05, Llama 3.2 1B tokenizer)

`kit/prompts/prompt_2048.txt` of `sarc-1.5-e2e-benchmark` (8192 bytes): 2049 tokens with llama.cpp's default
(it prepends the beginning-of-sequence token), **2048** with
`--override-kv tokenizer.ggml.add_bos_token=bool:false` (`llama-tokenize --no-bos` gives the same count).
Every llama.cpp run of this change passes that override and must report `2048 tokens`.
