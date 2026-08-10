# specs/041 addendum — texture-IO WMMA on M51, both drivers (2026-08-09/10)

Follow-on to the buffer sweep in `RESULTS.md`. Ports specs/040's texture-storage
coopmat (dev-igpu, production kernels, RADV-only) onto specs/041's **dbuf4 tsweep**
shaders, so the tile sweep can run with texture IO. Behind `ET_VK_TEXTURE_COOPMAT=1`;
unset, behavior is byte-identical to the buffer sweep.

## 1. Texture WMMA works and is numerically correct on M51 — first validation

specs/040 was 780M/RADV only and said "re-validate numerically on Xclipse before
trusting it there". Done:

| tile | `MMAS_PER_SG_M` | texture coopmat dispatched | correctness |
|---|---|---|---|
| `t32x64k32g22s32` | 1 (no dynamic indexing) | yes | 10/10 PASS |
| `t128x128k16g22s32` | **4 — dynamic `result[i][j]`** | yes (via rank-3) | **14/14 PASS**, incl. K=4096 |

The second row is the one specs/040 flagged as "the construct the Xclipse/AMD-PAL
compiler has broken before". It does **not** break, on either driver.

Two harness gaps had to be closed to get there, both of which would have produced a
falsely-green gate:
- The rank-2 correctness shapes **fall back to tiled at large tiles under texture IO**,
  so the gate never touched the coopmat path. Rank-3 cases (previously buffer-only)
  had to be enabled at texture3d — they are the only ones reaching `MMAS_PER_SG_M > 1`.
- texture+coopmat was labelled `unexpected_coopmat` (a pre-texture assumption in
  `run_linear_suite`), making the binary exit nonzero on correct runs.

## 2. A shared-memory bug that rebooted the board

The texture epilogue adds a `Csh` staging array on top of Ash/Bsh:
`SG_GRID_Y * MMA_M * WG_TILE_N` fp16. **That term is absent from
`tile_constraints`**, so tiles that are legal for buffer can exceed the 65536 B
shared-memory limit at texture IO — **20 of 160 tokens** (4 q4gsw, 16 dq8ca).

`tsweep_dbuf4_t128x256k32g18s32` needs **119808 B**. Rather than failing pipeline
creation, the driver hung the GPU and **rebooted the M51 mid-sweep**, invalidating
that whole run (all subsequent tokens returned nothing; the reboot also dropped root
and the clock pin).

Fixed in `QuantizedLinear.cpp`: the texture branch now computes the `Csh` budget
against `max_compute_shared_memory_size()` and rejects oversized tiles, so they fall
back to tiled. Verified: the killer token now dispatches
`linear_q4gsw_tiled_texture3d_texture2d_half` with no reboot. This protects any
caller, not just the sweep script. (A new `Adapter::max_compute_shared_memory_size()`
accessor was added; the limit was previously only printed in debug output.)

**Not fixed here:** `tile_constraints.derive` still models Ash+Bsh only, so its
offline legality answer remains wrong for texture IO. Left alone because changing it
would perturb the legal universe specs/036 swept over.

## 3. Driver B (`e0da99c1d1`) — texture IO does NOT rescue it

| path | driver A `f14c51b6f8` | driver B `e0da99c1d1` |
|---|---|---|
| buffer IO | works | **segfault**, 160/160 |
| texture IO, correctness | 14/14 PASS | **14/14 PASS** |
| texture IO, perf | all shapes | **1B only** (4/4); dies on 3B and 8B `w1_w3` |

The crash looks **cumulative, not shape-bound**: 1B `w1_w3` (2048→8192) passes while
3B `w1_w3` (3072→8192) fails at the same N. It survives ~2 heavy dispatches then
dies — consistent with a resource-tracking leak/UAF, which is exactly what
`e0da99c1d1` ("fix ResourceTracker not reaching HW via decorators") changes.

So no full 160-token A/B is possible under B even on texture.

## 4. The comparison that IS possible: 1B, 4 shapes, tile `t128x128k16g22s32`

PinnedMax, group_size 128, prefill M=2048, kernel-only GFLOP/s.

| shape (K,N) | A buffer | A texture | B texture | Btex/Atex | Btex/Abuf |
|---|---:|---:|---:|---:|---:|
| 2048, 2048 | 5625.8 | 4841.0 | 5226.6 | 1.080 | 0.929 |
| 2048, 512 | 4947.8 | 4139.8 | 4984.0 | 1.204 | 1.007 |
| 2048, 8192 | 5780.5 | 5089.1 | 5297.3 | 1.041 | 0.916 |
| 8192, 2048 | 5191.4 | 5141.4 | 5320.6 | 1.035 | 1.025 |
| **geomean** | **5376.0** | **4785.3** | **5205.4** | **1.088** | **0.968** |

- On driver A, texture costs **−11.0%** vs buffer — directionally consistent with
  specs/040's finding that the shipped geometries are buffer-tuned.
- **Driver B's texture path is +8.8% faster than driver A's**, winning all 4 shapes.
- B-texture lands within **−3.2%** of A-buffer, nearly erasing the texture penalty.

⚠️ Four shapes, one model, one tile. Indicative, not established. Plausibly the same
resource-tracking change that speeds this up is what makes the driver unstable.

## 5. What was NOT done

No texture tile sweep. The full 160-token texture run was attempted once and lost to
the reboot above; after the LDS fix it was not re-run. So specs/040's open question —
*is the optimal K-tile different for texture?* — remains open on M51. The −11.0%
texture deficit is measured only at the buffer-tuned geometry, which is precisely the
tuned-vs-untuned comparison specs/040 warned against reading as a verdict.
