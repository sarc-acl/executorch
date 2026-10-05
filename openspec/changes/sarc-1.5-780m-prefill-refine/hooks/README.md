# Hooks this change needs outside the dev zone

Neither patch is applied on the branch. `tools/build_hook.sh` applies both to a scratch copy of the tree and builds
there; every measurement of candidates 7 and 8 comes from such a build. Without a hook the dev-zone code that
depends on it compiles to nothing (`SARC_HAS_SDPA_SOFTMAX_OVERRIDE`, `SARC_HAS_SDPA_FUSED_HOOK`), so the branch
itself behaves exactly as without these candidates.

## `softmax-name-hook.patch` (candidate 7)

Release zone only: `impl/sarc/Select.h` (+3 lines) and `impl/sarc/SdpaCoopmat.cpp` (+2 lines).

- `Override` gains `std::string (*sdpa_softmax)(const std::string& sarc_name)`.
- `sdpa_softmax_shader_name()` passes the SARC softmax name through it when it is set.

The dev zone (`impl/sarc_dev/Overrides.cpp`, 780m block) then maps `ET_VK_SARC_780M_SOFTMAX=<tag>` to
`sarc_sdpa_attn_weights_softmax_780m_<tag>`. Tag `r3` is the candidate. It is only valid in front of a SARC attn*V
kernel (its zero fill stops where those kernels stop reading); the microbench pairing check fails any other pairing.
A promotion would not need the hook: the same change can be made to `glsl/sarc/sarc_sdpa_attn_weights_softmax.glsl`
directly, for the devices whose attn*V row is a SARC kernel.

## `sdpa-fused-hook.patch` (candidate 8)

Release zone: `impl/sarc/Select.h` (+19), `impl/sarc/SdpaCoopmat.{h,cpp}` (+14, +30). Upstream file already listed
in `sarc/HOOKS`: `impl/SDPA.cpp` (+7).

- `Override` gains `sdpa_fused_active(graph, q, input_pos)` and `sdpa_fused_add(graph, q, k, v, input_pos, out)`.
- `sarc::add_sdpa_fused()` is called at the end of `sdpa_impl` (LLM mode), after the three SDPA nodes; it calls
  `sdpa_fused_add`, which may append nodes that write `out`.
- `sarc::sdpa_fused_skip()` returns an empty launch geometry while `sdpa_fused_active` is true. It is consulted by
  `sarc::sdpa_gwg()` (QK^T and attn*V nodes) and by `pick_sdpa_softmax_gwg()` in `SDPA.cpp`, so for such a call
  the three kernels are not encoded (a zero-size dispatch is skipped by `DispatchNode::encode`).

The dev zone (`impl/sarc_dev/Sdpa780mFused.cpp`) supplies both functions. With
`ET_VK_SARC_780M_SDPA_FUSED=<variant per head_dim>` it appends the copy pass (`sarc_dev_780m_sdpa_kvt`) and the
fused kernel (`sarc_dev_780m_sdpa_fused3_*`) and reports a call as served when the tensors are fp16 buffers, the
head_dim has a variant, and S and `input_pos` are multiples of the variant's row tile and context block. Decode
(S = 1), unaligned prompts and `ET_VK_DISABLE_COOPMAT` keep the three kernels. The two attention-weight
temporaries are still allocated (they are needed whenever a call is not served).

What a promotion would have to decide, beyond moving the files: whether the fused kernel becomes a table row
(`Op::kSdpaFused` with its own fit check) instead of an override, and whether the copy pass should be replaced by
packing K and V into the tile layout once, where the cache is updated.
