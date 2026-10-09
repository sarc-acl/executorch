# sarc-1.5-b70-fused-port: the B580's fused attention kernel on the Arc Pro B70 (confirmation)

Device tag `b70`. Branch `topic/b70-fused-port`, forked from the first B70 campaign's pushed head
(`topic/xe2-prefill-refine` at `5617714b0`). **Parent of every comparison: `5617714b0` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine5`.** Host `fedora-gpu-eval`, card `b70-0` only (guest
PCI `0000:01:00.0`, `ETVK_DEVICE_INDEX=0`); the second card was not used. ANV, Mesa 26.2.3, kernel
7.2.9-200.fc44. Artifacts `/home/doremy/hmz-sarc-b70-fused/.artifacts/`. All of 2026-10-09, host clock, UTC.
Day detail and every table: `STATUS.md`.

## What this is

A confirmation, not a port: the fused attention kernel the B580 campaign built for the Intel matrix shape
(`sarc-1.5-b580-fused-port`, candidate 1 `b580-fused1`) is brought here unchanged, selected on top of this card's
final profile, gated and timed. No new kernel, no variant, no search, no sweep; one kernel screen over
already-built variants; one candidate.

**Result: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b70-fused1`, `GATE_PASS` (plain pass, next token SAME
everywhere), +8.26 % geomean over the parent on the build of the committed head (+8.34 % on the first gate), and
+72.87 % geomean over the first campaign's pristine parent.**

## What was brought, from where

Source repository `/mnt/linux-share/hmz-campaigns/b580-fused/executorch`, branch `topic/b580-fused-port`, **pinned
at `cea76c634`**. That branch is not on the public remote yet; the commits of this branch that carry its content
are marked "B580" below. Pushing this branch publishes them; that is intended (task section 6.5).

| commit here | what | source |
|---|---|---|
| `cfa31c1d2` | release-zone hook D4.1: `Override::softmax_variant` | B580: cherry-pick (`-x`) of `fab9606c3`, itself from `b969e8f1c2` (Orin campaign) |
| `cbbe36e0c` | release-zone hook D4.3: entry point of the fused attention node | B580: cherry-pick (`-x`) of `0ffc84a2d`, itself from `1c8861aa7e` (`topic/780m-prefill-refine`) |
| `087d4c4a9` | `test_llama_microbench.cpp`: the insert-only `4070ti-fused` blocks (99 lines added, none removed) | `origin/topic/4070ti-fused-port` at `ed8b5af91`, unchanged |
| `7d7877980` | `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.{glsl,yaml}`, `sarc_dev_b580_sdpa_kvt.{glsl,yaml}`, `impl/sarc_dev/b580/SdpaB580Fused.cpp` | B580: the files at `cea76c634`, byte for byte |
| `dfbaecca1` | `impl/sarc_dev/B70Sdpa.cpp`; the `b70-fused` blocks of `impl/sarc_dev/Overrides.cpp` (69 lines added, none removed) | new here |

- **Selector: the B580's file, reused unchanged**, not a `b70/SdpaB70Fused.cpp`. It is device-neutral for Intel:
  it asks for active SDPA rows and for the profile's variant list and names no device. The two functions it calls
  are defined in the `b70-fused` block here; a merge with the B580 branch meets them twice and has to join the
  two profile tables, which is the visible failure, where a second copy of the node would register silently on
  the one hook.
- **Base rows.** `Xe2Sdpa.cpp` activates its rows for `xe2-*` profile names only, so `B70Sdpa.cpp` registers the
  same two `kUnverified` rows for `bmg g31` while the profile name starts with `b70-`. Every other configuration
  selects what it selected before (`test_sarc_select`, below).
- **Profile.** `b70-fused1` is one name: `xe2-refine5` for every call the fused kernel does not take, plus
  `d64_t16x64s16m8g4roj` (head_dim 64) and `d128_t16x128s16m8g8oj` (head_dim 128), the pair `b580-fused1` is at
  `cea76c634`. (The task file names an earlier pair, `d64_t16x32s16m8ro` / `d128_t16x64s16m8ro`; that was the
  B580's first smoke definition and is not what it gated.) Six `b70-fused-<variant>` profiles exist for the screen.
- **Test blocks from the 4070 Ti port, not from the B580**: the B580 changed lines of the shared test file in
  place; here the file only gains delimited blocks (task section 4.3). The yaml has no generator.
- Both hooks are carried although nothing here sets `softmax_variant`: the fused hook's context is the softmax
  hook (alone it conflicts in `Select.h`), and with both the release zone differs from the parent by exactly the
  lines it differs by on the B580 branch.

The kernel reaches a prefill call when S and `input_pos` are multiples of 64 (head_dim 64) or 128 (head_dim 128):
the 2048-token prompt and the 1792-token `r1304.txt` qualify; the 1972-token `prompt_check.txt` and decode run
the three kernels of `xe2-refine5`.

### SPIR-V identity (`tools/spv_identity.sh`, `results/b70/identity/`)

On `topic1` (`8ba3607be`) and on `topic2` (`6ac44c483`, the committed head that was measured last):
`SPV_IDENTITY_OK`. All 1525 shaders of the parent build are byte-identical in the topic build (1572 shaders); all
47 `sarc_dev_b580_sdpa_fused*` and `..._kvt*` shaders are byte-identical to the B580 campaign's build `topic6`
(`247d08851`, on which its candidate 1 was gated). Shipped-SPIR-V golden: PASS, 53 variants, on `parent`, `topic1`
and `topic2`.

## Release-zone hooks (owner decision 2026-10-05, D4)

Two cherry-picks, each its own commit, unchanged. **The fused attention entry point (D4.3) is larger than a
switch and stays subject to the owner's review before any promotion.** Together: `impl/SDPA.cpp` +7,
`impl/sarc/SdpaCoopmat.cpp` +26 -1, `impl/sarc/SdpaCoopmat.h` +8, `impl/sarc/Select.h` +16. The diffs, as
`git show cfa31c1d2 cbbe36e0c` prints them:

**Softmax variant name, D4.1 (`cfa31c1d2`, cherry-pick of `fab9606c3`, from `b969e8f1c2`)**; nothing in this
campaign sets the field.

```diff
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
index d45f720bc..5a264e27e 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
@@ -202,7 +202,11 @@ std::string sdpa_softmax_shader_name(
   if (!device_has_active_rows(device_info(&graph), Op::kSdpaQk)) {
     return upstream_name;
   }
-  return "sarc_" + upstream_name;
+  const char* variant = get_override().softmax_variant;
+  if (variant == nullptr) {
+    return "sarc_" + upstream_name;
+  }
+  return "sarc_" + upstream_name + "_" + variant;
 }
 
 } // namespace sarc
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
index 93a9b8e96..166eba3e0 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
@@ -141,6 +141,9 @@ struct Override {
       const DeviceInfo& device,
       const ShapeInfo& shape,
       const std::optional<Choice>& table_choice) = nullptr;
+  // Suffix of a development softmax variant ("sarc_<softmax>_<suffix>"), or
+  // null: the release softmax.
+  const char* softmax_variant = nullptr;
 };
 void set_override(const Override& o);
 const Override& get_override();
```

**Fused attention node, D4.3 (`cbbe36e0c`, cherry-pick of `0ffc84a2d`, from `1c8861aa7e` of
`topic/780m-prefill-refine`).**

```diff
diff --git a/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp b/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
index c623a7235..c76a9c52f 100644
--- a/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
@@ -317,6 +317,9 @@ GlobalWorkGrid pick_sdpa_softmax_gwg(
     const std::vector<ArgGroup>& args,
     const std::vector<ValueRef>& resize_args) {
   (void)shader;
+  if (auto skip = sarc::sdpa_fused_skip(graph, resize_args)) { // SARC
+    return *skip;
+  }
   const SDPAMode mode = mode_of(resize_args);
   const ValueRef q = resize_args.at(0);
   // LLM reads H from axis -2, fused from axis -3 (handled by
@@ -815,6 +818,10 @@ void sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
       input_pos_symint,
       out,
       SDPAMode::LLM);
+
+  // SARC development hook: fused attention node (no-op without an override).
+  sarc::add_sdpa_fused(
+      graph, {q_projected, k_cache, v_cache, input_pos_symint, out});
 }
 
 void sdpa_with_kv_cache_impl(
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
index 5a264e27e..c6f9e9d3d 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
@@ -122,6 +122,9 @@ std::optional<GlobalWorkGrid> sdpa_gwg(
     ComputeGraph* graph,
     const vkapi::ShaderInfo& shader,
     const std::vector<ValueRef>& resize_args) {
+  if (auto skip = sdpa_fused_skip(graph, resize_args)) {
+    return skip;
+  }
   const std::optional<TileDims> dims = dims_for_kernel(shader.kernel_name);
   if (!dims.has_value()) {
     return std::nullopt;
@@ -143,6 +146,24 @@ std::optional<GlobalWorkGrid> sdpa_gwg(
       LocalWorkGroup(wg, 1u, 1u));
 }
 
+std::optional<GlobalWorkGrid> sdpa_fused_skip(
+    ComputeGraph* graph,
+    const std::vector<ValueRef>& resize_args) {
+  const Override& o = get_override();
+  if (o.sdpa_fused_serves == nullptr ||
+      !o.sdpa_fused_serves(graph, resize_args)) {
+    return std::nullopt;
+  }
+  return GlobalWorkGrid(
+      {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(64u, 1u, 1u));
+}
+
+void add_sdpa_fused(ComputeGraph& graph, const std::vector<ValueRef>& refs) {
+  if (get_override().sdpa_fused_add != nullptr) {
+    get_override().sdpa_fused_add(graph, refs);
+  }
+}
+
 vkapi::SpecVarList sdpa_qk_spec_vars(
     ComputeGraph& graph,
     const ValueRef q,
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h
index ab9d546fc..bc8a18fe0 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h
@@ -38,6 +38,14 @@ std::optional<GlobalWorkGrid> sdpa_gwg(
     const vkapi::ShaderInfo& shader,
     const std::vector<ValueRef>& resize_args);
 
+// Development hook (Override::sdpa_fused_*, null in a release). The empty
+// launch geometry for a QK^T / softmax / attn*V node whose call the fused
+// attention node serves, else std::nullopt; and the fused node itself.
+std::optional<GlobalWorkGrid> sdpa_fused_skip(
+    ComputeGraph* graph,
+    const std::vector<ValueRef>& resize_args);
+void add_sdpa_fused(ComputeGraph& graph, const std::vector<ValueRef>& refs);
+
 // Spec constants for the QK / AV nodes. `upstream` is release 1.5's list; it
 // is returned unchanged unless the device has an SDPA row, in which case the
 // SARC kernels' constants are appended after it.
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
index 166eba3e0..f1278114e 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
@@ -24,6 +24,9 @@
 #include <vector>
 
 namespace vkcompute {
+
+class ComputeGraph;
+
 namespace sarc {
 
 enum class Op {
@@ -144,6 +147,16 @@ struct Override {
   // Suffix of a development softmax variant ("sarc_<softmax>_<suffix>"), or
   // null: the release softmax.
   const char* softmax_variant = nullptr;
+  // Fused attention node (development; subject to the owner's review before
+  // any promotion). `sdpa_fused_add` may append nodes after the three SDPA
+  // nodes of an LLM-mode op, given the ValueRefs {q, k, v, input_pos, out}.
+  // While `sdpa_fused_serves` is true for a node's resize args, those nodes
+  // write the output and the three SDPA kernels dispatch nothing.
+  void (*sdpa_fused_add)(ComputeGraph& graph, const std::vector<int32_t>& refs) =
+      nullptr;
+  bool (*sdpa_fused_serves)(
+      ComputeGraph* graph,
+      const std::vector<int32_t>& resize_args) = nullptr;
 };
 void set_override(const Override& o);
 const Override& get_override();
```

With nothing selected nothing changes, shown on `topic1` and again on the final build `topic2`:

1. `test_sarc_select` from the parent export and from the topic export: release tables identical (1240 checks,
   31 rows); with the dev zone 1559 checks / 35 rows against 1561 / 37 (the two `b70-*` base rows).
   The four executables of the final check are kept in `.artifacts/raw/final-select/` (hashes in
   `results/b70/d4/final-select/hashes.txt`).
2. `spirv_golden.py` PASS on both builds.
3. Unmodified `verify.sh` with no environment against the parent's (`s0-parent-noenv`): `VERIFY_SAME`, 34 lines,
   rates removed, kernel names included, for `topic1` and for `topic2`.

Reproducibility from the committed branch (D4, second condition): `topic2` is an export of the head `6ac44c483`
with no local patch; `b70-fused1` dispatches the same kernels on it as on `topic1` (192 of 192 gated case runs
served by the two fused variants), its gate was run once more (`s3-final`: `GATE_PASS`) and so was the timed
session: +8.26 % against +8.34 % on `topic1`. Commits after `6ac44c483` touch only `openspec/`.

`sarc/tools/check.sh --no-build` at the head: PASS (zone rule, twin wrappers, `test_sarc_select` 1240 checks /
31 rows and 1561 / 37 rows; output in `results/b70/d4/check-no-build.txt`).

## Thresholds

`tools/thresholds.txt`: the rules were committed (`0160cf633`) before the first measurement, the calibrated
values (`4172bc183`) after the A/A and before candidate 1 was timed. `CLKMIN` 2457 MHz, 7 repeats (one A/A arm
spread 7.48 %), noise band +-2 %, baseline tolerance 3 %, kernel-screen margin 3 % in every round. Throttle
reasons of the xe driver: `thermal`, `prochot`, `ratl`, `vr_thermalert` reject a run; `pl1`, `pl2`, `pl4`,
`vr_tdc` are recorded only. Seen in the campaign's samples: `none` and `pl2`.

## Tools: what was changed from the first campaign's `tools/`

Copied from `sarc-1.5-xe2-prefill-refine/tools/`; the originals are not edited. Variables keep the `XE2_` prefix.

| file | change | reason |
|---|---|---|
| `host.sh` | `XE2_CARD` must be 0; `PARENT_COMMIT=5617714b0`, `PARENT_ENV`, `PRISTINE_COMMIT`; Python from the first campaign's venv, read only | task sections 1 to 3 |
| `host.sh` | workload pattern gains `comfy`, `nvtop`, `intel_gpu_top`; no monitor is exempt any more; `drm_clients`; `guarded` counts its polls per run; `foreign_wait` (30 minutes, then `FOREIGN_STOP`) | task sections 2 and 4.2 |
| `e2e5.sh` | a foreign GPU process is waited for before a run, and a run it disturbs is invalid and replaced, instead of ending the session; per run: guard polls, foreign engine time in the prefill window, DRM clients after the run, resident share of the model; model read into the page cache before each cell (D5); 6 extra pairs | task sections 2 and 4.2 |
| `sampler.py` | the B580 fused-port sampler (engine time of other DRM clients every 50 ms) for this card; a reading also without foreign clients | task section 4.2 |
| `gl.sh`, `gate.sh`, `parent_verify.sh` | `foreign_wait` before a GPU job; gate: repeats from `XE2_REPS`, `verify_diff.py` against the parent snapshot | as above; R7 |
| `sdpa_passes.sh` | hashes of the test binary, runner library and runner beside the pass statuses | task section 4.1 |
| `parent_verify.sh`, `screen_sdpa.sh`, `screen_rows.py`, `screen_sdpa_summary.py`, `summarize.py`, `sdpa_ref.sh`, `verify_diff.py`, `trace_attention.py` | taken from the B580 fused-port tools at `cea76c634` (snapshot with a name and an environment; screen resumable by saved rows with a reference profile; repeats from `env.txt`; reference error against the tuned parent; line diff; attention by kernel name), desktop idle wait removed | the parent here is a tuned profile too |
| `collect.sh` | names and paths; identity, D4 and chain files | |
| `spv_identity.sh`, `select_check.sh`, `chain1.sh` to `chain3.sh`, `thresholds.txt` (new) | SPIR-V identity, the kept `test_sarc_select` executables, the detached chains as run | task sections 4.5, 5, 6 |
| not carried over | the sweep tools, `gen_xe2.py`, the hook build and patch, `card_test.*`, `screen.sh`, `prof*.`, `test_guard.sh`, `test_hold.sh` | no search, no generation, no second card here; the two tests assert the two-card and abort behaviour this copy no longer has |

`test_export.sh` and `test_e2e5_guard.sh` pass on this copy; `test_nexttoken.sh` was started beside the first
build, waited on the build container as a foreign job (which is the new behaviour) and was stopped by hand; it
was not run again.

## The one screen (`results/b70/screens/screen1-select{,-runs}.csv`)

The B580's selection screen repeated over the same seven profiles, 3 rounds, cooled before every run, kernel time
per layer at S = 2048 in us (median of the rounds; every round is in `STATUS.md`):

| head_dim | profile | 1B | 3B | 8B | vs the parent's three kernels |
|---|---|---:|---:|---:|---|
| | parent `xe2-refine5` (QK^T + softmax + attn*V) | 1481 | 1285 | 1691 | 1.00x |
| 64 | `d64_t32x32s32m8ro` (the 780M's) | 4255 | | | 0.35x |
| 64 | **`d64_t16x64s16m8g4roj`** | **486** | | | **3.05x** |
| 64 | `d64_t16x64s16m8g4oj` | 551 | | | 2.69x |
| 128 | `d128_t16x64s32m8ro` (the 780M's) | | 9569 | 12577 | 0.13x |
| 128 | **`d128_t16x128s16m8g8oj`** | | **749** | **967** | **1.72x / 1.75x** |
| 128 | `d128_t16x64s16m8g4oj` | | 819 | 1047 | 1.57x / 1.62x |

No variant beats the B580's pair in any round; `b70-fused1` is the B580's pair.

## Results

Baseline and A/A (`s1-aa`): the parent re-measured within 0.22 % of the first campaign's `s12-final5` in every
cell (limit 3 %); parent against the topic build, both with the parent environment: -0.06 % geomean, largest cell
-0.22 %.

**Candidate 1, `b70-fused1`, against the tuned parent** (tok/s, median of 7 valid runs per arm, arms interleaved;
`s2-c1` on `topic1`, `s3-final` on the committed head `topic2`; recomputed from each `runs.csv`):

| cell | parent (`s3-final`) | `b70-fused1` (`s3-final`) | gain, committed head | gain, first gate (`s2-c1`) |
|---|---:|---:|---:|---:|
| 1B 4w | 17964.90 | 20686.90 | **+15.15 %** | +15.15 % |
| 1B 8da4w | 20480.00 | 24381.00 | **+19.05 %** | +17.86 % |
| 3B 4w | 7529.41 | 7968.87 | **+5.84 %** | +6.25 % |
| 3B 8da4w | 9570.09 | 9990.24 | **+4.39 %** | +4.39 % |
| 8B 4w | 3379.54 | 3482.99 | **+3.06 %** | +4.12 % |
| 8B 8da4w | 4481.40 | 4623.02 | **+3.16 %** | +3.16 % |
| geomean | | | **+8.26 %** | +8.34 % |

Every cell is outside the +-2 % band in both sessions. 84 timed runs in each, all valid: foreign engine time
0.00 %, at least one guard poll and 8 clock samples per run, lowest median clock 2533 MHz, no thermal reason.
The runner's timer steps by 1 ms (99 to 114 ms for 1B), so 1B medians move in steps of about 1 %.

Both gates `GATE_PASS` with 35 PASS lines and no FAIL line: SDPA tiers `all` / `extended` / `full`, 12 passes
each, every pass rc 0, 0 mismatches, `pairing=ok`, 192 of 192 case runs served by the fused kernel alone;
unmodified `verify.sh` identical to the parent snapshot line by line (`VERIFY_SAME`, 34 lines); next token parent
vs candidate SAME in all six cells on the three prompts; logits probe `PROBE_CHECK_OK`; `decide.py` `GATE_PASS`.
No next-token item differs, so the result is a plain pass and decisions D1 and D3 are not used. The kernel does
change the arithmetic; its error against the fp32 reference is smaller than the parent's on every production
shape (rms 2.0e-5 against 2.8e-5, maximum 7.1e-4 to 7.9e-4 against 1.06e-3 to 1.19e-3 at S = 2048;
`results/b70/sdpa-error/{c1-ref,final-ref}/`, identical in both runs) and on all eight `extended` cases; in one of
five synthetic `peaked` cases its maximum error is 8 % above the parent's with the rms 7 % below (reported, not a
gate item, fixed beforehand).

**Final stack against the first campaign's pristine parent** (`s4-pristine`: the build its `s12-final5` timed,
`6a7cc8cc6`, no profile, no `ET_VK_SARC_UNVERIFIED`, against `topic2` with `b70-fused1`; 7 valid runs per arm, 84
timed runs, all valid):

| cell | pristine parent | `b70-fused1` | total gain | pristine in `s12-final5` (citation) |
|---|---:|---:|---:|---:|
| 1B 4w | 11702.90 | 20686.90 | +76.77 % | 11770.10 |
| 1B 8da4w | 12337.30 | 24381.00 | +97.62 % | 12412.10 |
| 3B 4w | 4864.61 | 8000.00 | +64.45 % | 4864.61 |
| 3B 8da4w | 5251.28 | 9990.24 | +90.24 % | 5251.28 |
| 8B 4w | 2435.20 | 3512.86 | +44.25 % | 2435.20 |
| 8B 8da4w | 2730.67 | 4623.02 | +69.30 % | 2737.97 |
| geomean | | | **+72.87 %** | (first campaign: +59.23 %) |

The pristine arm reads within 0.6 % of the first campaign's own pristine arm. That session ends
`E2E5_INCOMPLETE` on one line: next token of 8B 8da4w against the pristine parent DIFFERs on `prompt_2048.txt` and
`prompt_check.txt`. These are the two items the first campaign's candidate 1 was accepted with (`ACCEPTED
(reference-error rule, owner decision 2026-10-04)`, its `proposal.md`); `prompt_check.txt` is not served by the
fused kernel at all, and against the tuned parent every item is SAME. Nothing new differs.

### Trace: attention before and after (warm ETDump, ms per 2048-token prefill, kernels by name)

| cell | arm | dispatch total | QK^T | softmax | attn*V | fused | K/V copy | attention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | pristine parent (`s4-pristine`) | 165.0 | 29.8 | 16.4 | 34.6 | | | 80.8 |
| 1B 4w | tuned parent `xe2-refine5` (`s3-final`) | 105.9 | 4.8 | 12.7 | 6.1 | | | 23.7 |
| 1B 4w | `b70-fused1` | 89.9 | | | | 7.3 | 0.24 | 7.5 |
| 8B 4w | pristine parent (`s4-pristine`) | 828.4 | 116.6 | 33.6 | 134.2 | | | 284.3 |
| 8B 4w | tuned parent `xe2-refine5` (`s3-final`) | 596.2 | 14.4 | 25.9 | 16.2 | | | 56.5 |
| 8B 4w | `b70-fused1` | 571.0 | | | | 31.4 | 0.69 | 32.1 |

All six cells of both arms: `results/b70/sessions/{s2-c1,s3-final}/attention.csv`. The gain is the attention
family alone: 23.7 -> 7.5 ms on 1B (-68 %), 37.2 -> 21.4 ms on 3B (-42 %), 56.5 -> 32.1 ms on 8B (-43 %); the
dispatch total falls by the same amount. Most of what is removed is the softmax's pass over the S x S matrix
(12.7 of 23.7 ms on 1B). After the change attention is 8 % of the 1B prefill and 6 % of the 8B prefill.

## B580 against B70

The B580 campaign's prediction for this card ("expect the same two variants"; kernel level 2.75x / 1.55x / 1.57x
for 1B / 3B / 8B; about +9 % geomean, most of it on 1B; numbers of `topic/b580-fused-port` at `cea76c634`, a
citation) held. The same pair wins the screen here with the same ranking in every row. At kernel level the B70
gains a little more than predicted, 3.05x / 1.72x / 1.75x, in the same proportion for all three models (the
B70's kernels are 1.3 to 1.5 times faster in absolute time: 486 / 749 / 967 us against 724 / 1139 / 1464). End
to end the two cards agree within 2.4 points in every cell: 1B +15.2 / +19.1 % here against the B580's +17.1 /
+19.0 %, 3B +5.8 / +4.4 % against +5.4 / +6.8 %, 8B +3.1 / +3.2 % against +3.5 / +4.3 %, +8.26 % geomean against
+9.18 %. The traces do not show a smaller saving here: the fused kernel removes 15 % of the 1B dispatch time
and 4.2 % of the 8B dispatch time on this card (16 of 106 ms, 25 of 596 ms), against 12 % and 3.6 % on the B580
by its trace (19 of 161 ms, 33 of 917 ms). The end-to-end difference between the cards is therefore not
explained by the kernel; it is of the size of the B580 session's own spread (that session ran on a desktop in
use, arm spreads up to 5 %) and of the 1 ms timer step on 1B here, and is left as measured. The reference
errors are identical to the B580's to every printed digit, as they must be for the same SPIR-V on the same
inputs. No difference between the two cards was found that would call for a card-specific variant (L7 again).

## Negative results and limits

- **Decode is 0 to 2.5 % slower with `b70-fused1`** (5 runs per arm, 32 tokens after the 2048-token prompt;
  `s2-c1`: 1B -2.5 / -1.2 %, 3B -1.3 / -1.5 %, 8B -0.8 / -1.0 %; `s3-final`: 1B -2.2 / 0.0 %, 3B -2.3 / -1.1 %,
  8B -1.1 / -0.9 %). The fused kernel does not serve decode; the cost was not located (no decode trace). Not a
  gate item; to be looked at before a promotion.
- No candidate 2: the B580 campaign gated its own (`b580-fused2`, the fp32 no-tail softmax for the calls the
  fused kernel does not take) at -0.13 % geomean and did not adopt it (its `STATUS.md` at `f613e3ed4`, read
  08:00 UTC, a citation), so by task section 6.4 none was brought here.
- The fused kernel serves only prompts whose length and position are multiples of the block (64 / 128 columns);
  the 1972-token real-text prompt runs the parent's kernels and gains nothing.
- Roofs were not re-measured: the campaign changes no linear kernel and the driver is the first campaign's
  (Mesa 26.2.3); its roofs of 2026-10-04 (`sarc-1.5-xe2-prefill-refine/results/xe2/roofline/`) stand as a
  citation, and no percent-of-roof is claimed here.
- What limits further progress on this card after this change: attention is 6 to 8 % of a prefill; the rest is
  the linear layers, which the first campaign's search left at 1 % steps.
- The B580 campaign's finding F1 (cooperative-matrix pipelines of the branch, shipped ones included, are SPIR-V
  1.3 without the full-subgroups flag the specification asks for) applies to this card unchanged; the fused
  kernel checks its subgroup layout at run time and every tier passed.
