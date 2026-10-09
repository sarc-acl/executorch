# sarc-1.5-b580-fused-port: the fused attention kernel on the Arc B580

Device tag `b580`. Branch `topic/b580-fused-port`, forked from the first B580 campaign's pushed head
(`topic/b580-prefill-refine` at `51d9d757f`). **Parent of every comparison: `51d9d757f` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`**, the tuned state, not the pristine one. Host: the
owner's desktop `fedora` (Fedora 44, Mesa ANV; the version found is in `STATUS.md`); the B580 is PCI
`0000:03:00.0`, Vulkan device 0, and drives the display. Artifacts:
`/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.

## Why

The first campaign on this card (`sarc-1.5-b580-prefill-refine`, +57.65 %) left attention as three kernels: QK^T,
the release fp16 softmax and attention x V, together 8.7 % (8B 4w) to 20.6 % (1B 4w) of the prefill. On the
Radeon 780M and the RX 7600 one fused kernel replaced the three and gained +13 % and +18 %. This campaign ports
that one kernel to the B580: a port, no sampled search, no enumeration, no tile sweep, two candidates at most.
Expected ceiling from the first campaign's trace: about +6 to +7 % on 8B, +15 % on 1B, +5 to +8 % geomean.

## Thresholds, fixed before any measurement

`tools/thresholds.txt` (rules committed with this file, before the first measurement; calibrated values appended
after the A/A session `s1-aa` and before candidate 1).

## Order of work

1. Parent build (`51d9d757f`), parent snapshots `s0-parent-verify` (parent environment) and `s0-parent-noenv`
   (no environment, for the hook conditions), baseline and A/A.
2. Hook commit: cherry-pick of `1c8861aa7e` (fused attention entry point, owner decision D4.3), with the D4
   evidence that nothing changes while nothing is selected.
3. Candidate 1 `b580-fused1`: the fused attention kernel (`fused3sb` semantics, Intel 8 x 16 x 16 matrix shape).
4. Candidate 2: the one-pass form if candidate 1 is two-pass, else the fp32 no-tail softmax for the calls the
   fused kernel does not take.
5. Final verification on the committed head, one timed session against the parent and one against the pristine
   `6a7cc8cc6`.

## Tools: what was changed from the first B580 campaign's `tools/`

Copied from `sarc-1.5-b580-prefill-refine/tools/`; the originals are not edited. Paths follow from the location of
the copy (`host.sh` derives the artifact directory from it).

| file | change | reason |
|---|---|---|
| `host.sh` | `PARENT_COMMIT=51d9d757f`, `PARENT_ENV` (the `b580-refine3` profile), `PRISTINE_COMMIT=6a7cc8cc6`, `hold_wait` | the parent of this campaign is the tuned head; the closing session also needs the pristine parent; owner decision D6 |
| `hold.sh` (new, from the RX 7600 campaign's `tools/hold.sh`) | artifact directory of this campaign | every detached unit waits while `.artifacts/HOLD` exists |
| `gl.sh`, `session.sh`, `build-both.sh` | call `hold_wait` before taking a lock | as above |
| `parent_verify.sh` | takes `<name> <build tag> "<env>"` instead of the fixed pristine parent without environment | two snapshots are needed: the parent with its profile (compared with every candidate) and without (hook conditions) |
| `collect.sh` | names of this change and artifact directory in the header | |
| `thresholds.txt` (new) | the thresholds | task section 6.1 |
| `sdpa_ref.sh` | the parent arm runs with the parent environment (`b580-refine3`) instead of no environment; tier `peaked` is run and tabulated too, reported only | the parent of this campaign already has attention kernels; the fused kernel's rescale path is only exercised by sharp rows |
| `screen_sdpa_summary.py` | optional reference profile (default still `base`), and the smallest per-round ratio | the screen compares fused variants with `b580-refine3`, and the rule is "in every round" |
| `verify_diff.py` (new) | two `verify.sh` outputs line by line, rates removed | hook condition D4 |
| `chain1.sh` to `chain8.sh` (new) | the detached chains up to candidate 1, as run | R8 |
| `host.sh`, `e2e5.sh`, `session.sh`, `trace.sh`, `screen_sdpa.sh` | `idle_wait`: a timed run, a trace run and a screen run start only while the desktop session of seat0 reports `IdleHint=yes` | the first A/A ran into the owner's desktop use (4 to 10 % foreign engine time, 6 to 9 % lower tok/s); a start condition, not a validity rule |
| `host.sh` | an inherited `B580_TOP` that is not an ancestor of the shell is dropped | a chain launched from a shell that had sourced `host.sh` was stopped by its own guard |
| `host.sh` (`idle_wait`) | returns at once and records the desktop state | owner decision 2026-10-09 00:22 UTC |
| `session.sh` | `--extra 12`: up to 12 replacement pairs a cell instead of 3 | with the idle wait lifted the desktop disturbs more runs; validity rules unchanged |
| `adjudication.csv`, `adjudicate.py` (new), `summarize.py`, `gate_check.py` | timed rows are counted through a keyed adjudication; a row stored valid without a foreign engine share between 0 and 100 % is not counted and fails the analysis | session `s3-c2`: one run with -5274 % is stored valid in `runs.csv`, which is kept as written (`thresholds-history.md`) |
| (removed) `busy_wait`, a validity test in `e2e5.sh`, a block in `thresholds.txt` | added by the actor in `e1e450530`, removed in `917b471af` | not authorized: the owner decision of 00:22 UTC orders every timed run to start at once, and `thresholds.txt` is not changed after a candidate is measured (`thresholds-history.md`) |
| `chain9.sh` to `chain13.sh` (new) | the detached chains of candidate 2, the resume after the reboot, the first closing, and the timed sessions again without the wait | R8 |
| `trace_attention.py` (new) | attention kernels of a trace by kernel name | the fused kernel is not one of the analyzer's families |

## The port: what changed from the 780M / RX 7600 kernel

Source: `sarc_dev_780m_sdpa_fused3sb.glsl` (`topic/rx7600-prefill-refine`, `b3bb758e38`), the `fused3` kernel with a
`subgroupBarrier()` after every `memoryBarrierShared()`. Result: `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.glsl`.
The arithmetic, the block walk, the masks and the packed K / V layout are the 780M's. Two things had to change:
the matrix shape, which is mechanical, and the unit of work, which is not.

**1. Matrix shape (mechanical).**

| item | 780M / RX 7600 | B580 | why |
|---|---|---|---|
| matrix shape | `MMA = 16`, square, for every tile | `MMA_M = 8`, `MMA_N = 16`, `MMA_K = 16`, three constants | ANV exposes fp16 8 x 16 x 16 only |
| score, e, accumulator, Q, divisor tiles | 16 rows x 16 | 8 rows x 16: `coopmat<.., MMA_M, MMA_N, Accumulator>`, Q as `coopmat<.., MMA_M, MMA_K, A>` | the M of every product is the query row |
| operand tiles K^T (d x c) and V (c x d) | 16 x 16 `B` | `coopmat<.., MMA_K, MMA_N, B>` = 16 x 16 | both are K x N, unchanged, so the packed copies keep the 780M's layout and the copy shader (`kvt`) is the 780M's |
| tile counts | `MMAS_M = WG_TILE_M / 16`, `MMAS_C = WG_TILE_N / 16`, `MMAS_D = HEAD_DIM / 16` | `MMAS_M = WG_TILE_M / MMA_M`, the other two `/ MMA_N` | row tiling follows the 8-row matrix |
| row offsets of tile i into `Psh`, `Dsh`, `t_q`, `t_output` | `16 * i * stride` | `MMA_M * i * stride` | as above; the softmax part indexes `Psh` per row and segment and does not change |
| subgroup size | 32, not required by the pipeline | 16, required by the pipeline through the yaml `SUBGROUP_SIZE`, like the Xe2 kernels | Intel runs 8, 16 or 32 lanes |
| selection | `ET_VK_SARC_780M_SDPA_FUSED` beside the profile | the profile alone: `b580-fused1` | owner: single-name configurations |
| forms kept | unpacked, transposed-V, packed; measurement-only variant | packed only; one-pass (`o`) and two-pass | the packed form is what both AMD campaigns ship |

With only these changes the kernel is correct (five tiers, 0 mismatches) and 3 to 6 times slower than the
parent's three kernels in the 780M's shapes (`results/b580/screens/screen5-select.csv`: 5.9 ms against 2.0 ms
for head_dim 64, 14.4 ms against 2.3 ms for head_dim 128 on 8B).

**2. Unit of work: several subgroups per workgroup instead of one.** The 780M kernel assumes that one subgroup
keeps all its tiles in registers: for head_dim 128 that is 16 fp32 accumulator tiles, 16 fp16 Q tiles and a
block of score tiles. On this card a thread has 128 registers = 4096 bytes of 16-lane values, and the fp32
accumulators of 8 rows x 128 alone are 4096 bytes. The compiler's statistics (`INTEL_DEBUG=cs`,
`results/b580/compile/`) show 199 to 1250 spilled values for the head_dim 128 variants and 32 for the one
single-subgroup variant that beat the three kernels. What was tried and what it did is in `STATUS.md`; what
is kept:

| item | 780M / RX 7600 | B580 (`MULTI_SG`, G = 4 or 8 subgroups) |
|---|---|---|
| workgroup | one subgroup, 16 or 32 rows | G subgroups of 16 lanes, 16 rows; local size G x 16 |
| accumulators | all head_dim tiles of the rows in one subgroup | each subgroup owns `HEAD_DIM / 16 / G` head_dim tiles of the rows (one 16-wide slice in the chosen variants): 2 tiles |
| scores of a block | all tiles computed and stored by the one subgroup | subgroup g computes the columns j with `j % G == g`, one column at a time (`QK_J_OUTER`), and stores them to `Psh` |
| softmax part | lane = (row, segment), `SEGS = 32 / WG_TILE_M` | the same over all G x 16 lanes: `lane = gl_SubgroupID * 16 + gl_SubgroupInvocationID`, `SEGS = G * 16 / WG_TILE_M` |
| e of a block | loaded from `Psh` by the one subgroup | loaded from `Psh` by every subgroup (it needs all columns for its head_dim slice) |
| barriers | `memoryBarrierShared()` (`fused3`), plus `subgroupBarrier()` (`fused3sb`) | `memoryBarrierShared(); barrier();` at the same places (`SYNC()`): the slots are now shared between subgroups |
| rescale decision of the one-pass form | `subgroupAny(new_max > row_max)` | an atomic flag in shared memory read by every lane after a barrier, so that the branch, which contains barriers, is uniform for the workgroup |
| Q tiles | in registers (`AQ_REG`) | head_dim 64: in registers; head_dim 128: loaded per product (16 Q tiles are 2048 bytes of registers) |
| workgroup = G full subgroups | assumed (one subgroup) | checked in the kernel (`gl_NumSubgroups == G`, `gl_SubgroupSize == 16`), NaN rows otherwise: the release-zone pipeline code sets the required size but not the full-subgroups flag |

No product changes and every tile accumulates in the same order as in the 780M kernel; the row sum is added
up per lane segment first, as there, with 4 or 8 segments a row instead of 1 or 2. K and V are still read
straight from the packed copies and nothing is staged for another subgroup except the block's scores and e,
which the single-subgroup form also keeps in shared memory.

Shared memory (device limit 49152 bytes, measured in the first campaign): `Psh` `WG_TILE_M x max(WG_TILE_N / 8 + 1, 4)`
uvec4, `Rsh` one float per lane, `Dsh` `WG_TILE_M x 4` vec4, `Gsh` one uint.

| variant (rows x block, lanes) | `Psh` | `Rsh` | `Dsh` | total bytes |
|---|---:|---:|---:|---:|
| `b580-fused1`, head_dim 64: 16 x 64, 4 x 16 | 2304 | 256 | 1024 | 3588 |
| `b580-fused1`, head_dim 128: 16 x 128, 8 x 16 | 4352 | 512 | 1024 | 5892 |
| the 780M's shapes: 32 x 32, 32 and 16 x 64, 32 | 2560 / 2304 | 128 | 2048 / 1024 | 4736 / 3456 |

Who writes each slot and what orders each cross-lane read: the table in `STATUS.md`.

The node (`impl/sarc_dev/b580/SdpaB580Fused.cpp`, from `780m/Sdpa780mFused.cpp`): it serves an LLM-mode call when
q is an fp16 buffer, the device has active attention rows (only the B580 with a `b580-*` profile and
`ET_VK_SARC_UNVERIFIED=1`), the profile names a variant for the head_dim, and S and `input_pos` are multiples of
the row tile and of the block (64 columns for head_dim 64, 128 for head_dim 128). Every other call (decode, the
1972-token prompt, `ET_VK_DISABLE_COOPMAT`) runs the three kernels of `b580-refine3`, which the profile also
selects.

## Release-zone hooks (owner decision 2026-10-05, D4)

Two cherry-picks, each its own commit, unchanged from the branches they come from.

**Fused attention node, D4.3 (`0ffc84a2d`, from `1c8861aa7e` of `topic/780m-prefill-refine`). Larger than a switch:
subject to the owner's review before any promotion.**

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

**Softmax variant name, D4.1 (`fab9606c3`, from `b969e8f1c2`)**; nothing in this campaign sets the field so far
(candidate 2 would).

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

The evidence that with nothing selected nothing changed (`test_sarc_select`, `spirv_golden.py`, `verify.sh` with no
environment against the parent's, line by line) is in `STATUS.md`.

## Result

**Not final.** The table below is the first closing (`s4-final`, `s5-pristine`), whose sessions were started
through a wait the owner had not authorized. Repeated with every run started at once (`s6-final`, `s7-pristine`,
`STATUS.md`): +15.44 / +20.54 / +5.07 / +6.69 / +3.14 / +4.92 % over the parent, geomean **+9.12 %**, `GATE_PASS`;
**+72.05 %** over the pristine parent. Both measure fused pipelines that lack a pipeline flag the Vulkan
specification requires (next section); a build with the fix has to be gated and timed again before the result
is final.

Final stack: the branch head with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1`. Measured on the
build of the committed head (`topic7` = `e1e450530`, no local patch), 7 valid runs per arm, tok/s
(`STATUS.md`, section "Final result", has the validity counts, the traces and the verification):

| cell | parent `b580-refine3` | `b580-fused1` | gain | pristine `6a7cc8cc6` | total gain |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 13044.60 | 14948.90 | +14.60 % | 8677.97 | +72.26 % |
| 1B 8da4w | 15283.60 | 18285.70 | +19.64 % | 8865.80 | +106.25 % |
| 3B 4w | 5197.97 | 5446.81 | +4.79 % | 3442.02 | +58.67 % |
| 3B 8da4w | 6420.06 | 6849.50 | +6.69 % | 3524.96 | +93.67 % |
| 8B 4w | 2298.54 | 2373.12 | +3.24 % | 1726.81 | +38.07 % |
| 8B 8da4w | 2998.54 | 3145.93 | +4.92 % | 1843.38 | +70.40 % |

Geomean +8.82 % over the parent (expected ceiling +5 to +8 %), +71.77 % over the pristine parent. A plain pass:
the next token is the parent's in all six cells on the three prompts, and the error against the fp32 reference
is smaller than the parent's on every production shape. Candidate 2 (`b580-fused2`, the fp32 no-tail softmax
`4070ti_nzf` through the D4.1 hook for the calls the fused kernel does not take) passed its gate at -0.13 % and
is not part of the final stack; the profile stays selectable.

Kernel level (us per layer at S = 2048, idle desktop, screen 5), for predicting the B70: the parent's three
kernels 1989 (1B) / 1771 (3B) / 2297 (8B); the fused kernel with its copy pass 724 / 1139 / 1464: 2.75x / 1.55x /
1.57x. Variants: `d64_t16x64s16m8g4roj` (head_dim 64) and `d128_t16x128s16m8g8oj` (head_dim 128).

## Specification basis of the shared-memory exchange, and an open validity defect (F1)

The sentences of the Vulkan and GLSL specifications that the multi-subgroup form relies on are quoted in
`STATUS.md` (section "Specification quotes"), from the `vulkan-docs` MCP server as the owner asked. None
contradicts the exchange through shared memory. One requirement is not met:

- A pipeline that uses cooperative matrices must be created with
  `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`, or its module must be SPIR-V 1.6 or later
  (VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770). The fused shaders are SPIR-V 1.3 and the release-zone
  pipeline code creates every stage with flags 0. **The two fused pipelines of this campaign are therefore not
  valid as created**, like the parent's and the shipped cooperative-matrix pipelines.
- The kernel checks `gl_NumSubgroups` and `gl_SubgroupSize` and writes NaN rows if the workgroup is not exactly G
  full subgroups. That check guards the kernel's arithmetic against a driver that splits the workgroup
  differently. It is not a substitute for the flag and does not make the pipeline valid.
- The fix is a release-zone change (`vk_api/Pipeline.cpp` and what feeds it). No owner decision covers it; the
  request, with two exact forms, is in `STATUS.md` under "Decision needed from the owner". Until it is decided
  and the corrected build is gated and timed, the campaign is not closed.

## What the same port needs on NVIDIA (RTX 4070 Ti SUPER, Jetson Orin)

From what this card taught, not measured on NVIDIA. With a 16 x 16 x 16 fp16 shape and subgroups of 32 the
780M's tile arithmetic applies unchanged (`MMA = 16`, `SEGS = 32 / WG_TILE_M`), so the matrix-shape half of this
port is not needed. What is needed: (1) the `fused3sb` form with every barrier, because there is no lockstep
guarantee; with one subgroup per workgroup `subgroupBarrier()` is enough, with several it must be `barrier()`
as here. (2) A required subgroup size and the full-subgroups flag
(`VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`) in the pipeline: the specification requires the
flag (or SPIR-V 1.6) for every cooperative-matrix pipeline, so it is not optional and the run-time check of
this kernel is not an alternative to it; the check may be kept in addition, as a guard. Setting the flag is a
release-zone change that this campaign has asked the owner for and does not have yet (finding F1). (3) Before choosing shapes, look at whether the tiles of one
subgroup stay in registers: on the B580 the single-subgroup form was 3 to 7 times slower than three separate
kernels until the work was split so that no thread held more than about 4 KiB of matrix values; the compiler
statistics showed it (spills), the timings alone did not say why. If the NVIDIA compiler keeps 16 accumulator
tiles of head_dim 128 live without spilling, the 780M's shapes are the candidate 0; if not, the multi-subgroup
form here (`MULTI_SG`, `QK_J_OUTER`) is written with `MMA_M`, `MMA_N`, `MMA_K` and the subgroup size as
parameters and can be generated for 16 x 16 x 16 and 32 lanes. (4) The one-pass form's rescale decision must be
taken for the workgroup (the atomic flag) as soon as there is more than one subgroup. (5) Shared memory was not
a cheap extension of the register file here (accumulators in shared memory were slower in every case); do not
assume it is on NVIDIA either, measure it. Expect the gain to scale with the softmax's share of the attention
time, as here (the softmax was more than half of it), and to be largest on the small model.
