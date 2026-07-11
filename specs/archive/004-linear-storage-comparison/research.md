# Research: Linear Shader Storage-Type Baseline Study

## Decision 1: Modify `test_llama_baseline_bench.cpp` in place, add a storage axis

**Decision**: Add a `kStorageTypes = {{"texture3d", utils::kTexture3D}, {"buffer", utils::kBuffer}}` axis to the existing `generate_cases()` cross-product (currently `kModels x kSchemes x kRegimes x op`), threading `storage` through `LinearConfig` and `make_case()` instead of the current hardcoded `const utils::StorageType storage = utils::kTexture3D;`. Add a `storage` column to the `RESULT,...` CSV line.

**Rationale**: The file already owns the per-model shape catalog (`kModels`, duplicated from `001`'s `results/shapes.json` since this binary has no JSON reader) and the case-generation/reporting pattern. Forking a second near-identical file would create two sources of truth for that catalog that could silently drift as models/shapes evolve. Modifying in place is the smaller, safer diff.

**Alternatives considered**: a new `test_llama_storage_bench.cpp` duplicating the catalog — rejected for the drift risk above; `001`'s original Texture3D-only numbers remain fully reproducible from the modified file (same cases, same shapes) so nothing is lost by extending rather than forking.

## Decision 2: `ET_VK_FORCE_TILED_LINEAR=1` is strictly required here, not just a formality

**Decision**: Every capture run for this feature sets `ET_VK_FORCE_TILED_LINEAR=1` for the whole process (a single env var, checked via `std::getenv()` at dispatch time in `can_use_q4gsw_coopmat()` — QuantizedLinear.cpp:175-176, the very first check).

**Rationale — verified directly in source before writing this plan, not assumed**:
`make_case()` constructs the input tensor as `ValueSpec input({cfg.M, cfg.K}, dt, storage, ...)` — a plain 2D tensor, no batch dimension. This is unlike the real e2e model, where `003` found the linear output is rank-3 (`[1, M, K]`) and fails `can_use_q4gsw_coopmat()`'s `dim_of(output) > 2` check independent of storage. In this microbenchmark, that check would **pass** (`dim_of == 2`). Checked the remaining eligibility conditions against this harness's actual shapes:
- `adapter->supports_cooperative_matrix()` / `subgroup_size() == 64` — hardware-level, same device as `001`, already confirmed true.
- Tile alignment (`GemmCoopmat.h:23-25`: `kCoopmatTileM=64, kCoopmatTileN=64, kCoopmatTileK=32`): prefill `M=2048` satisfies `M%64==0`; every one of the 8 ops' `N` values across all 3 models (checked: 4096/3072/2048 for wq/wo, 1024/512 for wk/wv, 14336/8192 for gate/up, 4096/3072/2048 for down, 128256 for lm_head) satisfies `N%64==0`; every `K` value satisfies `K%32==0`.
- `bias.set_none(true)` in `make_case()` — bias absent, satisfying that check too.
- dtype is `vkapi::kHalf` throughout.

Every single condition `can_use_q4gsw_coopmat()` checks (other than storage itself) is **already satisfied** by every prefill case in this harness. This means: without forcing the toggle, switching a prefill case's `storage` to `kBuffer` would not test "tiled dispatch at Buffer storage" — it would silently switch to the real coopmat dispatch, exactly the contamination Edge Case 1/FR-002 warns about. The toggle is not defense-in-depth here; it is the only thing standing between this study and silently measuring the wrong thing.

**Decode is unaffected either way**: `is_gemv_case` (checked in `pick_linear_qw_shader`, per `002`'s research) short-circuits before `can_use_q4gsw_coopmat()` is ever called, for both storage types, toggle or not. Decode's dispatch is always the `_coop` kernel regardless — consistent with FR-007.

**Alternatives considered**: relying on tile-misalignment or bias presence to naturally block coopmat for the `Buffer` cases — rejected once the alignment check above showed every real shape is already aligned; there is no case where this would work by accident.

## Decision 3: Significance determination — non-overlapping `mean ± 2·stdev` bands

**Decision**: For each (model, scheme, regime, op) case, call the storage-type difference a "real, reproducible effect" if the `Texture3D` and `Buffer` measurements' `[mean - 2·stdev, mean + 2·stdev]` intervals do not overlap; otherwise call it "within measurement noise."

**Rationale**: Reuses the same stdev-based reasoning this workstream already applies elsewhere (e.g. `005`'s target noise band), rather than introducing a new statistical test. Transparent and easy to verify by hand from the two reported numbers.

**Alternatives considered**: a formal Welch's t-test — rejected as unnecessary machinery for a go/no-go engineering read; the interval-overlap heuristic is simpler to explain in the report and consistent with how `001` already described its warm-up-effect finding (via direct comparison of measured ranges, not a formal test statistic).

## Decision 4: Cross-validate against `001`'s already-published `Texture3D` numbers

**Decision**: After capturing this feature's own `Texture3D` measurements (from the modified harness), compare them against `001`'s already-published `microbench` array entries for the same (model, scheme, regime, op) cases, within the same significance band as Decision 3.

**Rationale**: Confirms the harness modification (adding the storage axis, changing the CSV format) didn't accidentally change the existing Texture3D measurement itself — a real regression check on the tool, not just trust that a refactor was safe.

**Alternatives considered**: skipping this check since the code change is "just adding a parameter" — rejected; this workstream's established discipline (per the constitution and `001`'s own mid-implementation correction) is to verify empirically rather than assume a code change is behavior-preserving.
