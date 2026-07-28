# specs/040: Cooperative-matrix linear on TEXTURE storage

**Date**: 2026-07-27 / 28
**Branches**: `dev-try-texture-wmma` (off `yanwen/dev-1.3` @ `ffbfbeb0d0`) — original work;
`yanwen/dev-igpu` @ `f5d26322e3` — ported, texture support only, tiles untouched.
**Device**: AMD Radeon 780M (RADV PHOENIX), Mesa 25.2.7, Vulkan 1.4.318, subgroup_size 64.
**Protocol**: Llama 3.2 1B, 2048-token prefill (`p2048_exact.txt` + `--num_bos=1`;
`prompt_tokens: 2048` verified on every run), `--max_new_tokens=8 --warmup=true
--temperature=0`, 5 reps. Kernel-level numbers from ETDump, 112 `linear_q4gsw_*` dispatches.

---

## 1. Summary

1. **A cooperative-matrix (WMMA) linear can read and write texture storage.** Implemented,
   numerically correct, and dispatching on the 780M. It needs no `.pte` re-export and no
   clone/island transition nodes — a plain texture or default-storage pte gets WMMA.
   Against the tiled path it replaces: **~2.45×** (742–765 → 1864–1891 tok/s).
2. **Nobody had ever tried it.** The `kBuffer` gate was inherited policy, not a design
   decision (§3).
3. **Whether it beats whole-graph-buffer WMMA depends entirely on `WG_TILE_K`** (§6) — this
   is the most transferable finding here. The K-tile that is optimal for buffer is *not*
   optimal for texture. On the 780M the fastest config overall remains **buffer + K32**.
4. **A missing `d16` fold was worth 9.3% of the texture kernel** (§7). Worth checking first
   on any half-precision texture path.
5. Two pre-existing bugs surfaced along the way (§9), one of them an 8.2× regression on this
   GPU that invalidates any 4w coopmat number measured on `yanwen/dev-1.3` since `f20ef0c965`.

Everything is behind `ET_VK_TEXTURE_COOPMAT=1`. Unset, behavior is byte-identical to before.

---

## 2. Why this was thought impossible, and why it isn't

`coopMatLoad` / `coopMatStore` take a **pointer**. An image is a **handle** with no element
linearization — no `OpAccessChain` yields one. So a cooperative-matrix instruction can never
name a texel. That much is a SPIR-V type-system fact, not a driver limitation.

But it is not on the critical path, because the workaround is already universal in this tree:

- **All 44 `coopMatLoad`s in the repo already load from `Workgroup` (shared) memory** — zero
  from global. Verified by disassembling the shipped SPIR-V. The global → LDS staging hop
  exists everywhere, so on the input side only the *spelling of the global fetch* changes.
- Only the result store went straight to the SSBO. And
  `sdpa_compute_attn_weights_coopmat.glsl:252-276` already had the
  `coopMatStore` → LDS → copy-out pattern (it needs it to apply the causal mask to an opaque
  accumulator). That is exactly the shape the texture epilogue needs.
- Coopmat coexisting with `sampler3D` in one shader was *already shipping*:
  `linear_dq8ca_qw_coopmat.glsl:113-114` declares two, and the weight has been
  `texture2d`-or-`buffer` parameterized all along.

---

## 3. The gate was inherited, not decided

Searched all 28 active specs, 14 archived specs, git log on all 7 local branches, and
stashes: **no prior attempt and no explicit rejection.** The buffer requirement enters the
record exactly once, as an observation —
`specs/archive/003-wmma-shader-candidates/results/wmma-candidates-report.md:15`: *"output
tensor storage is TEXTURE_3D; can_use_q4gsw_coopmat() requires Buffer storage"*. Every later
spec treated "export to buffer" as the fix.

Three signs it was never a deliberate choice:

- The upstream commit that introduced coopmat dispatch (`40f4fa756c`, D103971112 / PR #19009)
  has an **empty commit body**.
- `GemmCoopmat.h:34-40` documents "three device-capability gates beyond shape alignment" and
  lists 2D-outputs-only, `subgroup_size()==64`, `!is_integrated_gpu()`. **Storage is not among
  them**, yet `graph.storage_type_of(out) == utils::kBuffer` sits silently in the boolean.
- In `QuantizedLinear.cpp` the storage check is a bare three-line `if` with **no comment**,
  between two checks carrying 12 and 6 lines of rationale.

---

## 4. Implementation

Three files. Opt-in via `ET_VK_TEXTURE_COOPMAT=1`.

**`linear_qw_coopmat.yaml`** — new `IO_STORAGE` parameter (default `buffer`) and two variants,
`linear_q4gsw_coopmat_texture3d_{texture2d,buffer}_half`. Variant names must match what the
C++ builds: `add_storage_type_suffix(output_storage)` + `add_storage_type_suffix(weight_storage)`
+ dtype.

**`linear_qw_coopmat.glsl`**

- `t_output` / `t_input` declared with `IO_STORAGE`; codegen expands to
  `writeonly image3D` (rgba16f) and `sampler3D`.
- A-operand staging factored into `load_a_vec4()`. The buffer index `row * K4 + k_hv4` and the
  texel `(k_hv4, row, 0)` are **the same address in two notations** — a width-packed texture3d
  holds elements `[4x, 4x+3]` of row `m` at texel `(x, m, 0)`, and width-packing groups 4
  K-consecutive halves, the same direction the A tile is already read contiguously.
- New `#ifdef IO_TEXTURE` epilogue: `coopMatStore` into `shared float16_t Csh[]`, barrier,
  then `imageStore`. `Csh` holds `SG_GRID_Y` bands of `MMA_M` rows (4 KB at 128×64) so at
  iteration `i` every subgroup drains its own accumulator row-block concurrently into
  disjoint global row ranges. A full `WG_TILE_M × WG_TILE_N` staging buffer would blow the
  LDS budget (32 KB at a 128×128 tile). Cost: `MMAS_PER_SG_M` × 2 barriers.

**`QuantizedLinear.cpp`** — `can_use_q4gsw_coopmat()` gains `allow_texture_io`, passed `true`
only from `pick_linear_qw_shader`. This matters: the dq8ca path calls the same function and
has **no** texture variants, so relaxing it globally would resolve to a missing shader.
Texture is accepted only when input **and** output are both `kTexture3D` and both
width-packed, because one `IO_STORAGE` param covers the pair.

### Port to `yanwen/dev-igpu`

The glsl was byte-identical between the two branch bases, so it was copied wholesale. The
yaml and cpp deltas were applied by hand, **excluding all tile changes** — dev-igpu's
`WG_TILE_N 64 / WG_TILE_K 32`, `kQ4gswCoopmatDims = {128, 64, 32, 128}` and
`kDq8caQ4gswCoopmatDims = {128, 64, 32, 256}` are its own 780M sweep results (specs/035, 038)
and are left untouched. `dev-try-texture-wmma`'s `WG_TILE_N 128→64` revert is *not* carried
over — that was a workaround for a dev-1.3-only problem (§9).

Note: adding yaml variants requires a **re-configure**, not just a rebuild — the shader-lib
file GLOB and the generated-output list are evaluated at configure time.

---

## 5. Correctness

| check | result |
|---|---|
| `test_llama_microbench --correctness-only` | **48 PASSED / 0 FAILED** on both branches |
| …of which dispatch `linear_q4gsw_coopmat_texture3d_*` | **8 cases** |
| `spirv-val`, all 4 variants | PASS |
| e2e 8-token continuation, all arms | byte-identical |
| decode (M=1, gemv) | works — falls to the pre-existing `linear_q4gsw_coop_texture3d_texture2d_half` |

`generate_correctness_cases()` already sweeps `{kTexture3D, kBuffer}`, so the new path is
covered for free wherever `M % 128 == 0` and `N % 64 == 0`:

```
M128_K128_N128   M128_K128_N256   M128_K256_N128   M256_K128_N64
M256_K128_N128   M256_K256_N256   M128_K2048_N128  M128_K4096_N128
```

The K=2048/4096 rows matter most: fp16 accumulator error grows with the reduction length, so
a shader change can pass at small K and diverge at production K.

**Coverage gap** — `generate_correctness_cases()` runs its rank-3 cases at Buffer only, on the
stated assumption that *"Texture3D+rank-3 exercises the pre-existing tiled path"*. This work
breaks that assumption: the real exported model's activations are rank-3 `[1, M, K]` and now
go through texture coopmat. Rank-3 texture is currently covered **only** by the e2e
byte-identical continuation. Adding those rows is the obvious next commit.

### Dispatch proof

The microbench cannot prove dispatch here — `generate_linear_perf_cases()` hard-assigns
`kTexture3D → tiled`, `kBuffer → coopmat`, so it never builds a texture coopmat perf case.

Two independent proofs instead. **Negative control** — the tsweep yaml has no texture3d
variants, so requesting one must fail by name:

```
ET_VK_TEXTURE_COOPMAT=1 ET_VK_Q4GSW_COOPMAT_VARIANT=tsweep_t128x64k16g22s32 llama_main ...
  → Could not find ShaderInfo with name
    linear_q4gsw_coopmat_tsweep_t128x64k16g22s32_texture3d_texture2d_half
```

**ETDump**, which also confirms every gate condition on the real graph:

```
kernel_name: linear_q4gsw_coopmat_texture3d_texture2d_half   ×112 dispatches
  args: sizes [1, 2048, 2048] / [1, 2048, 8192], dtype Half,
        storage TEXTURE_3D (in and out), packed_dim 0
```

112 = 16 layers × 7 linears (q/k/v/o/gate/up/down), none missed.

---

## 6. THE HEADLINE: optimal K-tile is storage-dependent

ETDump, `linear_q4gsw_*` only, 112 dispatches, identical shapes. The only variables are IO
storage and `WG_TILE_K`:

| `WG_TILE_K` | texture | buffer | texture / buffer |
|---|---:|---:|---:|
| **16** (dev-1.3 base) | 671.1 ms | 708.2 ms | **0.948** — texture 5.2% *faster* |
| **32** (dev-igpu, HW-swept) | 656.9 ms | **575.2 ms** | **1.142** — texture 14.2% *slower* |
| K16 → K32 | **−2.1%** | **−18.8%** | |

**The K32 retile is a buffer-specific optimization.** It buys buffer 18.8% and texture
essentially nothing, which is enough to flip the verdict between the two storages.

Mechanism (plausible, **not** confirmed with performance counters): deepening the K-step
doubles the per-chunk global load burst. For buffer that means fewer, larger, better-amortized
bursts. For texture the load *instruction* count doubles as well, while each `image_load`
still carries only 64 bits of payload — consistent with the texture path being **issue-limited
rather than latency-limited**, so a deeper K gives it nothing to amortize.

End-to-end, same 5-rep protocol:

| arm | K16 (dev-try-texture-wmma) | K32 (dev-igpu) |
|---|---:|---:|
| tex-tiled | 764.8 | 761.8 |
| tex-wmma | 1847.5 | 1863.8 |
| embq-tiled | 750.6 | 742.3 |
| embq-wmma | 1852.7 | 1890.8 |
| buf-wmma | 1808.7 | **2057.1** |

All CoV ≤ 1.0%.

**This comparison is not fair to texture and must not be read as a verdict on the approach.**
The buffer tile has been e2e-swept twice (specs/035, specs/036); the texture variant has never
been swept even once. It is borrowing a geometry chosen for a path with different memory
behavior. This is tuned-vs-untuned.

**Standing conclusion for the 780M: buffer + K32 is the fastest config (2057.1 tok/s).** The
value of texture WMMA on this GPU today is that it reaches 1864–1891 tok/s on a texture /
default-storage pte with **no re-export**, versus 742–762 for the tiled path it replaces.

---

## 7. The `d16` fold — 9.3% hiding in a type declaration

A half `sampler3D` is *typed* to return `vec4` (fp32). Writing
`const vec4 v = texelFetch(...)` and then `packHalf2x16` makes ACO emit a plain `image_load`
into 4 VGPRs **plus a `v_cvt_pk_rtz_f16_f32` per pair to narrow back** — the hardware performs
an fp32 widening nobody asked for, and the shader spends instructions undoing it.

Consuming the fetch as `f16vec4` immediately lets ACO fold the narrowing into the instruction
as `image_load ... d16`: packed fp16 straight into 2 VGPRs, zero conversion. Lossless either
way — the source is rgba16f. This is the same shape the tiled texture path already used
(`linear_fp_input_tile_load.glslh`), which is why *it* got d16 and this shader didn't.

Final ISA, texture coopmat shader:

| | before | after (K16) | after (K32, dev-igpu) |
|---|---:|---:|---:|
| `image_load` | 10 | 10 (**8 with `d16`**) | 20 (**16 with `d16`**) |
| VGPRs per A fetch | `v[12:15]` = 4 | `v[12:13]` = 2 | 2 |
| `v_cvt_pk_rtz_f16_f32_e32` | 16 | **0** | **0** |
| `v_wmma_f16_16x16x16_f16` | 8 | 8 | 16 |
| `ds_store_b16` / `image_store` | 16 / 1 | 16 / 1 | 16 / 1 |

Effect on K16: linear kernel **739.9 → 671.1 ms (−9.3%)**, e2e 1748.7 → 1847.5 tok/s.
Correctness re-verified after the change (48/48, `spirv-val` PASS, continuation still
byte-identical).

**Check for a missing `d16` first whenever a half-precision texture path looks slow.**

---

## 8. What the ISA says about texture vs buffer

Disassembly of the two shipped variants (`spirv-dis` + `RADV_DEBUG=shaders`, K16 tile):

| | texture | buffer |
|---|---:|---:|
| `v_wmma_f16_16x16x16_f16` | 8 | 8 |
| `ds_load_b128` (LDS → MMA) | 8 | 8 |
| `image_load` | 10 | 2 |
| `buffer_load_b128` | 2 | 6 |
| `v_cvt_pk_rtz_f16_f32_e32` | 0 (post-d16) | 0 |
| `ds_store_b16` | 16 | 0 |
| `image_store` | 1 | 0 |
| `buffer_store_b16` | 0 | **64** |
| `OpCooperativeMatrixLoadKHR` ptr class | 6 × `Workgroup` | 6 × `Workgroup` |
| `OpCooperativeMatrixStoreKHR` ptr class | 2 × `Workgroup` | 8 × `StorageBuffer` |

Three things follow.

**The compute core is identical.** Same MMA count, same LDS loads, same
`Workgroup`-class `coopMatLoad`s. From LDS → registers → MMA the two variants are the same
program; only the global↔LDS edges differ.

**Input costs texture 2× the instructions.** 8 `image_load` where buffer needs 4
`buffer_load_b128` for the same bytes. An rgba16f texel carries **64 bits** of payload;
`buffer_load_b128` carries **128**. This is inherent to 4-component 16-bit texels and cannot
be optimized away.

**Output actually favours texture.** Buffer's `coopMatStore` lowers to **64 scalar
`buffer_store_b16`** — 2-byte global stores. The texture path's forced LDS round-trip yields
16 `ds_store_b16` plus wide `image_store`. **Lead for the shipped buffer path**: staging the
buffer output through LDS too may be a free win. Not attempted.

**On specialized texture hardware.** The 780M does have a dedicated texture path — MIMG-class
instructions, 256-bit image descriptors loaded by `s_load_b256` (vs 128-bit for buffer), and
`dim:SQ_RSRC_IMG_3D unorm dmask:0xf` addressing fields with no buffer equivalent. What it
does *not* provide is any benefit for this access pattern: its advantages are swizzled 2D
addressing and filtering, while GEMM wants payload width per instruction, where texture is
structurally behind. The claim that image and buffer share the same L0/L1/L2 hierarchy is
**architectural belief, not verified here** — it would need RGP / `RADV_THREAD_TRACE`.

---

## 9. Pre-existing bugs surfaced

**The shipped 128×128 q4gsw coopmat tile is 8.2× slower on the 780M.** Same binary, same
buffer pte, only `ET_VK_Q4GSW_COOPMAT_VARIANT` differing:

| tile | 1B 2048-prefill |
|---|---:|
| `yanwen/dev-1.3` default 128×128/2×2/sg32 (`f20ef0c965`, specs/036's M51 winner) | **220.8** |
| prior 128×64/2×2/sg32 (`tsweep_t128x64k16g22s32`) | **1817.2** |

The unmodified `dev/executorch/cmake-out-vk` binary (built 2026-07-23, before the retile)
returns 1817.21 — bit-identical, which also confirms the `load_a_vec4` refactor is a no-op on
the buffer path. specs/036 e2e-ranked that tile on M51/Xclipse 970 and it wins there; nobody
re-checked the 780M. **Any 4w coopmat number measured on the 780M with `yanwen/dev-1.3` at or
after `f20ef0c965` is running an 8× pessimized kernel.** `yanwen/dev-igpu` is unaffected. The
tile default needs to become per-device.

**`[[unroll]]` is not honored on the drain loop.** The disassembly shows 2 static
`coopMatStore` + 3 `OpLoopMerge` (vs 8 + 1 for buffer) — the `barrier()` in the body keeps the
loop rolled, so `result[i][j]` **is** dynamically indexed, contrary to the intent. RADV/ACO
accepts it and it validates numerically, but coopmat arrays are opaque per-lane storage and
dynamic indexing is exactly the class of construct the Xclipse/AMD-PAL compiler has broken
before (three separate codegen bugs are already documented in these shaders). Check this first
if the texture variants ever miscompile on another driver. Hand-expanding the drain so each
`i` gets its own barrier removes the risk.

---

## 10. Scope and limits

- ~~**4w (`q4gsw`) only.**~~ **Superseded 2026-07-28 — see §13.** The dq8ca A operand is
  indeed structurally buffer, but that turned out not to block anything: only the output
  needed converting.
- **One device.** `image3D` writes inside a coopmat shader are novel in this tree.
  Re-validate numerically on Xclipse before trusting it there.
- **Prefill only, 1B only.**
- **The texture variant has never been tile-swept.** Every texture number here is at a
  geometry chosen for buffer.
- Cross-device extrapolation of the storage verdict is out of scope by request; the 780M is
  one of the more buffer-favoring devices in the fleet (specs/037 `buf_x` 4w = 1.21), so these
  numbers should not be read as representative of phones.

---

## 11. Open items

1. **Texture-specific tile sweep** — the single highest-value follow-up, and the only way to
   make §6's comparison fair. specs/036's machinery exists but its tsweep yamls generate
   buffer variants only; `IO_STORAGE` needs threading through first.
2. **LDS-stage the buffer output too** (§8) — possible free win on the *shipped* path.
3. **Rank-3 texture correctness rows** in `generate_correctness_cases()` (§5).
4. **Hand-expand the drain loop** to eliminate the dynamic coopmat indexing (§9).
5. **Make the coopmat tile per-device** so the 780M stops running the M51 geometry (§9).
6. Confirm the issue-limited hypothesis in §6 with performance counters rather than inference.

---

## 12. Reproduction

```bash
cd /home/doremy/sarc-acl/dev-igpu/executorch
source .venv/bin/activate
cmake -S . -B cmake-out-vk                      # re-configure: new yaml variants
cmake --build cmake-out-vk -j16 --target install
cmake -S examples/models/llama -B cmake-out-vk/examples/models/llama \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON           # dev-igpu wires etdump on this flag
cmake --build cmake-out-vk/examples/models/llama -j16
cmake -S backends/vulkan/test/custom_ops -B cmake-out-vk/backends/vulkan/test/custom_ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/cmake-out-vk
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j16
```

Then `ET_VK_TEXTURE_COOPMAT=1` on any 4w texture / default-storage pte. Correctness:
`ET_VK_TEXTURE_COOPMAT=1 test_llama_microbench --correctness-only`. Attribution:
`--etdump_path=X.etdump`, then aggregate by the `kernel_name` field of each event's JSON name
(skip wrapper events — `Method::execute`, `DELEGATE_CALL`, `ETVK_*` — they nest and would
double-count). **ETDump `perf_data` raw values are milliseconds here.** ISA:
`RADV_DEBUG=shaders`.

---

## 13. Addendum 2026-07-28 — dq8ca (8da4w) texture WMMA

§10 scoped dq8ca out. That was wrong about the consequence, not the premise: its A operand
*is* structurally buffer, but the shader **never reads `t_input` at all** — activations come
from `t_packed_int8_input`, and the binding comment has said `int_input_sums(3 - unused)` all
along. Only the output needed converting, so the same `IO_STORAGE` split applies, with the
`Csh_out` / `imageStore` epilogue ported from `linear_qw_coopmat`. `t_input` is still declared
from `IO_STORAGE` so the binding layout agrees with the graph's actual tensor storage.

`linear_dq8ca_q4gsw_coopmat_texture3d_texture2d_half` dispatches on all three models
(112 / 196 / 224 dispatches for 1B / 3B / 8B, ETDump-confirmed).

**Performance is parity, not a win.** Unlike q4gsw — whose tiled path costs 2240 µs and had
room to give — the dq8ca tiled dot4 kernel was already efficient:

| 1B, 112 dispatches | tiled | coopmat |
|---|---:|---:|
| `linear_q4gsw_*` | 2240 µs | 653 µs |
| `linear_dq8ca_q4gsw_*` | 565 µs | 560 µs |

e2e 1B 8da4w embq: 1919.9 → 1932.6 tok/s (+0.7%). This reproduces specs/038's buffer-path
verdict (dq8ca coopmat ≈ tiled) on texture storage.

**The useful result is numerical.** Because the coopmat shader derives its sums in-shader and
never touches the scratch buffer that specs/040-dq8ca-input-sums-oob undersizes, it is exempt
from that bug. Measured at M=2048 on 1B embq, 10 reps per cell, release build:

| | 4w | 8da4w |
|---|---|---|
| WMMA | 1844.4 ✅ | 1932.6 ✅ |
| tiled | 761.4 ✅ | 1919.9 ❌ wrong token |

Same pte, same binary — only the dispatched kernel differs. 3B behaves identically; 8B's tiled
arm happens to be correct because its allocation exactly meets the requirement at M=2048 (it
breaks at 2049). **So the input-sums OOB is a tiled-path defect, not a blanket dq8ca one**, and
coopmat is currently the only way to get a fast *and* numerically correct 8da4w long-prompt
prefill.

Caveat: that is an e2e greedy-argmax check, not op-level validation — the same distinction
that let Mali-G710 pass e2e while failing the op test with `nan`.
`generate_correctness_cases()` has no texture-coopmat dq8ca case yet; adding one belongs with
the rank-3 texture rows in §11.3.

Full 3-model 2×2 matrix (12 cells × n=10, interleaved with per-round arm rotation, release
build with devtools OFF; the event-tracer build was re-run in full as a control and agrees
within ±0.65% with mixed sign): 4w's WMMA gain is 2.42× / 2.65× / 2.85× on 1B / 3B / 8B and
grows with model size, while dq8ca's is 1.01–1.04×. 8da4w's lead over 4w therefore collapses
from 2.5–3.1× on the tiled path to 1.05–1.15× once 4w has WMMA, and that residue *widens* with
model size rather than converging.
