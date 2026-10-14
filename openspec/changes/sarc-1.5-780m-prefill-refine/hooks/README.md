# Release-zone hooks (owner decision 2026-10-05)

Until 2026-10-05 candidates 7 to 10 were measured through two local patches that were never committed
(`softmax-name-hook.patch`, `sdpa-fused-hook.patch`, applied to a scratch tree by `tools/build_hook.sh`; all three
are in the history up to `cafd851db`). The owner decision of 2026-10-05 ("allow all", `CAMPAIGN.md`) permits
committing a small, inert entry point for each. They are now two commits on this branch and the local patches are
gone. Nothing else in the release zone changes: no shipped shader, no table row, no golden, no tolerance, no tool.

With no `ET_VK_SARC_780M_PROFILE` (and no `ET_VK_SARC_780M_SDPA_FUSED`) both entry points are null and every
device adds the same nodes, picks the same shaders and launches the same geometry as before.

## 1. Softmax variant name (`b969e8f1c`)

`common/hooks/softmax-variant-hook.patch` of the Orin campaign, applied unchanged with `git am` (the diff is
byte-identical to the supplied file), so that the topic branches merge without a conflict. 8 lines:
`Override::softmax_variant` in `impl/sarc/Select.h` (null by default) and `sdpa_softmax_shader_name()` in
`impl/sarc/SdpaCoopmat.cpp`, which returns `sarc_<upstream name>_<variant>` when it is set.

Dev zone: a profile of `ET_VK_SARC_780M_PROFILE` names the variant (`impl/sarc_dev/Overrides.cpp`, 780m block; the
environment variable `ET_VK_SARC_780M_SOFTMAX` no longer exists). The variant shaders were renamed to the hook's
form, e.g. `sarc_sdpa_attn_weights_softmax_buffer_half_780m_r3` (before:
`sarc_sdpa_attn_weights_softmax_780m_r3_buffer_half`); their sources did not change. Variant `780m_r3` is only
valid in front of a SARC attn*V kernel (its zero fill stops where those kernels stop reading): the profile does not
set it under `ET_VK_DISABLE_COOPMAT`, and the microbench pairing check fails any other pairing.

A promotion would not need this hook: the same change can be made to
`glsl/sarc/sarc_sdpa_attn_weights_softmax.glsl` directly, for the devices whose attn*V row is a SARC kernel.

## 2. Fused attention node (`1c8861aa7`): subject to the owner's review before any promotion

49 added lines (the local patch had 70): `impl/sarc/Select.h` +13, `impl/sarc/SdpaCoopmat.h` +8,
`impl/sarc/SdpaCoopmat.cpp` +21, and `impl/SDPA.cpp` +7 (an upstream file already listed in `sarc/HOOKS`).

- `Override::sdpa_fused_add(graph, {q, k, v, input_pos, out})` and
  `Override::sdpa_fused_serves(graph, resize_args)`, both null by default.
- `sarc::add_sdpa_fused()` is called once at the end of `sdpa_impl` (LLM mode), after the three SDPA nodes.
- `sarc::sdpa_fused_skip()` returns an empty launch geometry while `sdpa_fused_serves` is true. It is consulted by
  `sarc::sdpa_gwg()` (QK^T and attn*V nodes) and by `pick_sdpa_softmax_gwg()` in `SDPA.cpp`, so for such a call
  the three kernels are not encoded (a zero-size dispatch is skipped by `DispatchNode::encode`).

What was moved out of the release zone to get there: the check that the node is in LLM mode and the unpacking of
the arguments are now in the dev zone, and the feature macros are gone.

Dev zone: `impl/sarc_dev/780m/Sdpa780mFused.cpp` supplies both functions. It is in a subdirectory because
`sarc/tools/check.sh` compiles `impl/sarc_dev/*.cpp` into the GPU-free `test_sarc_select`. It hands the pair to the
780m block of `Overrides.cpp` (`register_sdpa_fused_780m`), which keeps it and applies it from whichever static
initializer runs last: the shared `Registrar` of `Overrides.cpp` starts from an empty `Override`, so a registration
made only from the other file is lost when that file is initialized first (found on the first build of this form:
the softmax variant was selected and the fused node was not; `<artifacts 10-05>/raw/wt1-smoke/`). With a profile
that names fused kernels (or `ET_VK_SARC_780M_SDPA_FUSED=<variant per head_dim>`) it appends the copy pass
(`sarc_dev_780m_sdpa_kvt`) and the fused kernel (`sarc_dev_780m_sdpa_fused3_*`) and reports a call as served when
the tensors are fp16 buffers, the head_dim has a variant, and S and `input_pos` are multiples of the variant's row
tile and context block. Decode (S = 1), unaligned prompts and `ET_VK_DISABLE_COOPMAT` keep the three kernels. The
two attention-weight temporaries are still allocated (they are needed whenever a call is not served).

What a promotion would have to decide, beyond moving the files: whether the fused kernel becomes a table row
(`Op::kSdpaFused` with its own fit check) instead of an override, whether the copy pass should be replaced by
packing K and V into the tile layout once, where the cache is updated, and the wording of the `SDPA.cpp` line in
`sarc/HOOKS` (unchanged here).

## Profiles (`ET_VK_SARC_780M_PROFILE`, on top of `ET_VK_SARC_DEV_PROFILE=780m-refine3`)

| profile | 4w kernel per shape | softmax | attention |
|---|---|---|---|
| `refine9`, `refine10` | yes | release | three kernels |
| `softmax-r1` | no | `780m_r1` | three kernels |
| `c7` (candidate 7) | no | `780m_r3` | three kernels |
| `c8` (candidate 8) | no | `780m_r3` where the fused node does not serve | fused, two-pass (`...rk`) |
| `c9` (candidate 9) | `refine9` | same | fused, one-pass (`...rko`) |
| `c10` (candidate 10) | `refine10` | same | fused, one-pass |

The measurement-only softmax `780m_m1` (no exp, wrong results by design) has no profile.
