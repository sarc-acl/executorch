# Port to upstream main: pr/wmma-families (2026-10-16, local, not pushed)

Branch `pr/wmma-families` (worktree `topic/wmma-families/pr-port`) on upstream `main` 062003504c, from the
family rows of `fae9e53e76` (Xe2 and NVIDIA only; no per-device rows, no dev hooks). 36 files, +4785/-15:

- `748e9bfdae` vk_api: full-subgroups flag (F1), Vulkan memory model, driver ID / vendor ID /
  cooperative-matrix property list / max shared memory accessors.
- `452fec82f6` family key and rows (`impl/gemm/linear/qw/CoopmatFamily.{h,cpp}`, pure C++).
- `53049bc694` q4gsw path (`q4gsw/Q4gswLinearWmma.*`, shaders `linear_q4gsw_wmma[_bx]_<tile>`).
- `7accdda872` dq8ca path (`dq8ca_q4gsw/Dq8caQ4gswLinearWmma.*`, `linear_dq8ca_q4gsw_wmma[_rowmajor]_<tile>`,
  packer `quantize_and_pack_4w_with_group_sums`).
- `dd36029886` tests: host unit test `vulkan_coopmat_family_test` (10 gtests, llvmpipe key -> no family),
  `etvk_bench` set `q4gsw_linear:wmma_correctness`.

No `sarc` in the diff or file names. All 18 ported variants compile to SPIR-V byte-identical to the fork's.

## Checks

| Device | wmma_correctness | upstream correctness | Family kernels dispatched |
|---|---|---|---|
| Arc B580 | 24/24 | 102 passed, 26 skipped (upstream's dq8ca fp16 skips) | all four Xe2 rows |
| Arc Pro B70 | 24/24 | all passed | flib, g44fli buffer, xe2bto i and n13 |
| RTX 4070 Ti SUPER | 24/24 | all passed | bt, bx, t128x128k16g22s32, rowmajor bf |
| Jetson Orin | 24/24 | all passed (26 skipped, as upstream) | bt, bx, t128x128k16g22s32, rowmajor bf |
| llvmpipe | 24/24 | identical to the upstream baseline, same kernel lists | none (no family) |

Jetson Orin: `etvk_bench` cross-built from the pr-port tree with the Orin campaign's image
(`et-jetson-cross:jp7.2.1`, `nvidia-orin/tools/jetson-cross/aarch64.cmake`, the pinned glslc).

B580 end to end, Llama 3.2 1B prefill tok/s (median of 3, port / upstream): 512 tokens 4w 10449 / 3241,
8da4w 13474 / 6737; 511 tokens 4w 2762 / 3297, 8da4w 6636 / 6636; 5 tokens 4w 135 / 74, 8da4w 250 / 238.

## Open review questions (before any upstream PR)

1. NVIDIA rows fire only on the measured Llama 1B/3B/8B projections (a model-specific shape table).
2. The Xe2 8da4w K*N >= 24e6 threshold comes from one device.
3. Key reach: Xe2 requires ANV (Intel Windows driver excluded); the NVIDIA key matches other NVIDIA
   generations with the same shapes (unmeasured).
4. Unaligned 4w prompts on a family device get slower (511 tokens: -16 % vs upstream), because a family-built
   op falls back to `linear_q4gsw_tiled`/`coop` instead of upstream's current 4w kernels (the fork's
   behaviour too).
5. The row-major 8da4w kernel reads activations in whole 128-row tiles past an M-row buffer for unaligned M
   (inherited); pad the temporary buffer to a multiple of 128 rows.
6. K = 14336 texture3d images exceed ANV's reported 2048 extent (known caveat, N13.7).
7. Whether to add `wmma_correctness` to the A10G line of `.github/workflows/vulkan.yml`.

Deviations from the fork: dq8ca family path also requires `can_use_int8_dot_product()`; unused shader
features, `kBufBuf` and `csh_full`/`csh_pool` dropped; the packer keeps only texture3d-input variants; no
`test/op_tests` gtests (libtorch not in the container); bench tolerance abs 0.25 (1.0 for K > 8192) on
zero-mean data.
