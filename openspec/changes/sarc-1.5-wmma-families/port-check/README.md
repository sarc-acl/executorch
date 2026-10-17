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
| Radeon 780M (RADV, no family) | 24/24 | 204 passed / 26 skipped, coopmat_correctness 64/64; equal to the upstream baseline binary run on the same device | none (no family kernel name in any log) |
| llvmpipe | 24/24 | identical to the upstream baseline, same kernel lists | none (no family) |

Jetson Orin: `etvk_bench` cross-built from the pr-port tree with the Orin campaign's image
(`et-jetson-cross:jp7.2.1`, `nvidia-orin/tools/jetson-cross/aarch64.cmake`, the pinned glslc).

The other members of the proposal's eight devices (RX 7600, RX 7900 XTX, Xclipse M51) are company-side and
not checked; they have no family in this port, so they run upstream's path.

B580 end to end, Llama 3.2 1B prefill tok/s (median of 3, port / upstream): 512 tokens 4w 10449 / 3241,
8da4w 13474 / 6737; 511 tokens 4w 2762 / 3297, 8da4w 6636 / 6636; 5 tokens 4w 135 / 74, 8da4w 250 / 238.

## Open review questions (before any upstream PR)

1. NVIDIA rows fire only on the measured Llama 1B/3B/8B projections (a model-specific shape table).
2. The Xe2 8da4w K*N >= 24e6 threshold comes from one device.
3. Key reach: Xe2 requires ANV (Intel Windows driver excluded); the NVIDIA key matches other NVIDIA
   generations with the same shapes (unmeasured).
4. ~~Unaligned 4w prompts on a family device get slower (511 tokens: -16 % vs upstream), because a family-built
   op falls back to `linear_q4gsw_tiled`/`coop` instead of upstream's current 4w kernels (the fork's
   behaviour too).~~ Fixed 2026-10-17 (`fc3adb58c6`), see below.
5. ~~The row-major 8da4w kernel reads activations in whole 128-row tiles past an M-row buffer for unaligned M
   (inherited); pad the temporary buffer to a multiple of 128 rows.~~ Fixed 2026-10-17 (`f5ed8a271d`) in the
   kernel instead of padding, see below.
6. K = 14336 texture3d images exceed ANV's reported 2048 extent (known caveat, N13.7).
7. Whether to add `wmma_correctness` to the A10G line of `.github/workflows/vulkan.yml`.

Deviations from the fork: dq8ca family path also requires `can_use_int8_dot_product()`; unused shader
features, `kBufBuf` and `csh_full`/`csh_pool` dropped; the packer keeps only texture3d-input variants; no
`test/op_tests` gtests (libtorch not in the container); bench tolerance abs 0.25 (1.0 for K > 8192) on
zero-mean data.

## Fix of review questions 4 and 5 (2026-10-17, local, not pushed)

`pr/wmma-families` now `f5ed8a271d` (two commits on `dd36029886`):

- `fc3adb58c6` q4gsw texture-IO kernels (`linear_q4gsw_wmma`, `_bx`) take a partial last M tile: rows past M
  read activation row M - 1 and are not stored. Selection treats M >= 128 as M rounded up to a multiple of 128
  (`kWmmaPartialTileM`), so an unaligned M gets the row of the next aligned M. M < 128, buffer IO and 8da4w
  keep the alignment rule and the old fallback.
- `f5ed8a271d` row-major 8da4w kernel: activation rows, scales and zero points of rows past M read row M - 1,
  rows past M are not stored; nothing is read past the M-row tensors. No padding of the temporary buffer
  (it would also have depended on that buffer never being resized).

Tried and dropped: storing rows past M onto row M - 1 (removes the drain branch, but bit-equality of those rows
is not guaranteed and row M - 1 is the next-token row); gating the drain check on a workgroup-uniform
partial-tile flag and hoisting the clamped A rows out of the K loop (no measurable gain).

Checks (logs in `fix-2026-10-17/`; host 1B/8B PTEs from the NAS, Orin 1B copied to the host):

| Device | unit test | wmma_correctness (+4 unaligned 4w cases) | correctness | coopmat_correctness |
|---|---|---|---|---|
| host | 11/11 (new `PartialMTile`) | | | |
| Arc B580 | | 32/32 | 102 passed, 26 skipped (as before) | 32/32 |
| Arc Pro B70 | | 32/32 | 102 / 26 | 32/32 |
| RTX 4070 Ti SUPER | | 32/32 | 102 / 26 | 32/32 |
| Jetson Orin | | 32/32 | 102 / 26 | 32/32 |

Unaligned cases select the partial-tile family kernel on texture IO (Xe2 `flib`; NVIDIA `t256...bt`,
`t128...g22`, `bx...f32c`) and the old kernels at M = 127 and on buffer IO.

Prefill tok/s, upstream / port before / port after (`dd36029886` / `f5ed8a271d`):

| Device, model | 511 tokens | 300 | 129 | 512 |
|---|---|---|---|---|
| B580, 1B 4w (one run) | 3024 / 2542 / 10429 | 2655 / 2174 / 8824 | 1500 / 1817 / 5375 | 3066 / 10449 / 10667 |
| B580, 8B 4w | 536 / 371 / 2129 (one run) | | | port 2124 / after 2124 (median of 3) |
| 4070 Ti, 1B 4w (median of 3) | 6636 / 5495 / 21292 | 5172 / 4478 / 14286 | 2804 / 2633 / 7167 | 6649 / 22261 / 21333 (23 vs 24 ms, timer step) |
| 4070 Ti, 1B 8da4w (median of 3) | 7406 / 30059 / 30059 | 6122 / 23077 / 23077 | 3909 / 11727 / 12900 | 7420 / 32000 / 32000 |
| 4070 Ti, 8B 8da4w (median of 3) | port 6724 / after 6724 | | | port 6737 / after 6737 |
| Orin, 1B 4w, after only (3 runs) | 1352-1363 | 1103-1119 | | 1369 |

Outputs: B580 8da4w (no change on Xe2) identical in all three arms at every length; 4w at 512 identical
in all three arms, at unaligned lengths the fixed build differs from both others on the degenerate repeated
"the" prompt (different kernels; first generated token the same), and on a natural 357-token prompt the fixed
build's 60 tokens equal the old port's. 4070 Ti: 8da4w after == before at every length (1B and 8B); 4w natural
prompt identical in all three arms. Orin natural prompt gives a sensible summary on both quantizations; the
first 4w run at 357 tokens had a one-off 158 tok/s (first use of that pipeline), later runs about 1210.

Kernel cost at aligned shapes (etvk_bench GPU timestamps, before vs after, interleaved runs):
- 4070 Ti (8 runs): 4w +0.4 to +0.8 %; 8da4w row-major within +-0.4 % except K = 14336 at +1.5 to +2 %
  (reproduced; about +0.5 % without the drain check). End to end 8B 8da4w unchanged (6724 / 6737 tok/s).
- Orin (6 runs, spread <= 0.3 %): 4w -2.6 to +0.5 %; 8da4w row-major +0.4 to +0.7 %.
- B580 (8 runs, workstation load about 12 from other sessions, spread up to 4 %): 4w texture mixed
  -5.5 to +4.3 % (K = 14336 +4.3 %); buffer and the unchanged 8da4w kernels within +-2 %. End to end 8B 4w
  at 512 tokens unchanged (2124 tok/s). Repeat on an idle workstation before quoting B580 kernel numbers.
