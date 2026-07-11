# Research: End-to-End Texture3D vs. Buffer Storage Comparison

## Decision 1: Restore the dropped `default_storage` check (dead-code fix, not a hack)

**Decision**: Fix `backends/vulkan/utils.py`'s `TensorRepSet.make_tensor_repr()`
(lines 964-987) so that, for tensors whose repset is ambiguous (both storage
types valid — the case for linear/matmul activations per `op_registry.py`'s
`CONTIGUOUS_ANY` registration, lines 396-421), it consults the caller's
storage preference instead of unconditionally returning `TEXTURE_3D`
(currently hardcoded at line 979-981, commented *"Prefer texture storage
over buffer storage"*).

**Grounding — investigated directly before writing this plan, not assumed**:
A `storage_type_override` mechanism already exists, fully wired end to end:
a `CompileSpec` key (`vulkan_preprocess.py:97-100`) → `VulkanPartitioner`
constructor option (`vulkan_partitioner.py:286-317`) →
`TagMemoryMetaPass(default_storage_type=...)` → stored as
`self.default_storage` (`tag_memory_meta_pass.py:150,155`). But
`self.default_storage` is **never read anywhere** today — `git log -p` on
`bedce91e7f4795869158b96ef479d92317b13871` ("Rewrite Memory Metadata Tagging
Pass", PR #12927, 2025-07-31) shows a prior version of `make_tensor_repr()`
(or its equivalent) *did* consult it before choosing a default; that check
was dropped in the rewrite. So this is a genuine regression, not a
theoretical gap — the mechanism was designed to do exactly what this
feature needs and stopped working by accident.

**Why this is safe to restore, verified not assumed**: `default_storage_type`
defaults to `VkStorageType.TEXTURE_3D` (`tag_memory_meta_pass.py:150`) —
identical to today's hardcoded behavior. Restoring the check changes
**nothing** for any existing caller, since nothing today has any path to
request a different value (`export_llama_lib.py` never forwards
`storage_type_override` to `VulkanPartitioner` at all). It only enables a
new, explicitly opt-in path.

**Alternatives considered**: writing an entirely separate, parallel
storage-forcing mechanism (e.g. a new pass) — rejected; restoring the
already-designed-for-this mechanism is smaller, safer, and removes latent
dead code as a side benefit.

## Decision 2: New CLI flag, mirroring `--vulkan-force-fp16` exactly

**Decision**: Add a new export flag (e.g. `--vulkan-storage-override
{texture3d,buffer}`) to `examples/models/llama/export_llama_lib.py`,
forwarded through `extension/llm/export/partitioner_lib.py`'s
`get_vulkan_partitioner()` to `VulkanPartitioner`'s `storage_type_override`
compile option — the same plumbing shape already used for
`--vulkan-force-fp16` → `force_fp16` → `VulkanPartitioner`.

**Rationale**: Precedented, minimal-diff, consistent with existing
conventions; a future contributor extending this pattern again (e.g. for
memory layout) has one clear example to follow, not two divergent ones.

**Alternatives considered**: a standalone script calling the export
pipeline's internal Python functions directly, bypassing
`export_llama_lib.py`'s CLI — rejected; `export_llama_lib.py` already owns
model loading, quantization, tokenizer handling, and SDPA fusion, and
reimplementing that path outside the CLI would risk silently diverging from
`001`'s already-validated export recipe for the `Texture3D` baseline.

## Decision 3: Smoke-check design (not a correctness re-verification)

**Decision**: Per the Clarifications session, the `Buffer`-storage export's
smoke-check is: the `llama_main` runner completes without error/crash for
the fixed prompt at `--temperature 0`, and produces the expected
`generated_tokens` count with non-empty, non-degenerate text (not, e.g., a
single token repeated for the entire generation length — a common failure
signature for a badly broken model). This reuses `001`'s existing
`PyTorchObserver` stats line (`generated_tokens`, etc.) already emitted by
every e2e capture — no new instrumentation needed.

**Rationale**: Directly matches the Clarifications outcome — catch this
feature's own export/config mistakes (crash, garbage), not re-verify
Texture3D-vs-Buffer numerical equivalence (assumed, per Clarifications, as
an existing ExecuTorch/Vulkan-backend guarantee).

**Alternatives considered**: comparing generated text token-for-token
against the `Texture3D` variant — rejected per the Clarifications session
explicitly (that would be re-verifying numerical equivalence, out of scope).

## Decision 4: Buffer size-limit risk — try and report, not pre-block

**Decision**: Since `within_buffer_limit()` (`utils.py:702-713`) is dead code
(never called — confirmed by grep) and no active pre-export size check
exists, this feature does not attempt to add one. Each configuration's
export/run is attempted; if a specific tensor (most plausibly `lm_head`,
`N=128256` at prefill `M=2048` → a multi-hundred-MB fp16 buffer) fails at
allocation or dispatch, that configuration is reported as blocked per
FR-006 with the actual error, not estimated or worked around.

**Rationale**: Real RDNA3/AMD hardware typically supports buffer sizes well
beyond the informal `128MiB` constant in code (which is unenforced and may
not reflect an actual hard device limit) — whether this is actually a
problem is an empirical question this feature answers by trying, consistent
with this workstream's "verify empirically" discipline, not by assuming
either outcome.

**Alternatives considered**: pre-computing every tensor's byte size and
skipping any configuration whose largest tensor exceeds `128MiB` — rejected;
this constant is unenforced/possibly stale, and skipping without trying
would just be a different kind of unverified assumption.

## Decision 5: Reuse `001`'s e2e capture methodology and schema exactly

**Decision**: The `Buffer`-storage e2e capture uses the identical procedure
`001` already established (fixed 2048-token prefill / 1024-token decode,
`--temperature 0`, 5 repeated runs, discard cold-start/warm-up drift,
steady-state mean±stdev, strict no-concurrent-load discipline) and writes to
the same `e2e` JSON schema `001`/`005` already use
(`prefill_tokens_per_sec`, `decode_tokens_per_sec`, `variance`, etc.).

**Rationale**: Only reusing the exact same methodology and schema makes the
`Texture3D` (already captured) vs. `Buffer` (this feature) comparison
apples-to-apples — a new or divergent methodology would reopen exactly the
"is this a fair comparison" question `004` already had to answer once.
