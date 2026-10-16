# Device check of the family rows (2026-10-16)

Commit `fae9e53e76` (family key and the Xe2 / NVIDIA family rows). Release tables only: no `ET_VK_SARC_*`
variable set, so every choice comes from the family rows (and the RTX 4070 Ti SUPER per-device rows).
`sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` on each member, on the device lock of
its campaign. Binaries: x86-64 from `sarc/tools/build.sh --llama` (one binary for the B580, B70 and
4070 Ti), aarch64 from the Orin campaign's cross recipe (`nvidia-orin/tools/parent_build.py`, image
`et-jetson-cross:jp7.2.1`); hashes in `binaries.sha256`. SPIR-V golden PASS (61 shipped variants) on both
builds; every promoted variant is byte-identical to the dev-named variant the matrices measured.

| Member | Dispatched (texture3d / buffer) | Production diff | Next token | Compared with |
|---|---|---|---|---|
| Arc B580 | 4w flib 12 / g44fli 12; 8da4w xe2bto i 8+8, n13 4+4 | 12/12 PASS | 3 SAME, 1 DIFFER (pre-existing, below) | s9-xe2-family-b: same kernels and counts |
| Arc Pro B70 | as B580 | 12/12 PASS | 3 SAME, 1 DIFFER (pre-existing, below) | xe2-family-b stage: same kernels and counts |
| RTX 4070 Ti SUPER | 4w bt 11, ga 1 (K = 14336 override) / ga 1+11; 8da4w bf 12 / zpgtr g44 12 | 12/12 PASS | 4 SAME | s2-gate-nv-family-b: bx replaced by the ga override at K = 14336, as intended; same status lines |
| Jetson Orin | 4w bt 11, bx 1; 8da4w bf 12 / stock (q4gsw w_4x8, dq8ca tiled) | texture3d 6/6 PASS, buffer 6/6 FAIL | 4 SAME | verify-control and verify-orin-fused-nv-family-b: identical status lines |

- Orin buffer IO keeps the stock kernels (no rows, before and after); their production diff and the
  microbench correctness item fail identically in the campaign's control. Orin's kernel list line in
  `verify.out` does not parse (also in the control); the dispatched names are from `orin.linear-*.json`.
- `linear ... rc=1` on all members is the microbench's own status, identical in the campaign controls.
- B580 and B70, 1B 8da4w unaligned prompt (r1304): default vs tiled DIFFER. A/B on both cards
  (`llama_main`, one new token): the default output of this build, of the pre-change build
  (`topic/wmma-families` merge trial) and of the measured dev profile (family-b) are byte-identical; the
  tiled baselines of both builds are identical to each other; with the dev profile (Xe2 SDPA in both arms)
  default and tiled are SAME. So the difference is the tiled baseline under the stock SDPA, present before
  this change; the family rows change no output.
