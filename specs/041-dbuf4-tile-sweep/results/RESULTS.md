# specs/041 results — dbuf4 tile sweep, microbench, M51 primary, 2026-08-09

Prefill M=2048, geomean of kernel-only GFLOP/s over the 12 real Llama linear
shapes. PinnedMax (980000/5333000/934000, sysfs-verified). group_size=128.
Harness `test_llama_microbench` md5 `9625ac9501a1560c8ddc874a7f07b353`.

## Driver B (`e0da99c1d1`) — UNUSABLE on M51, zero tokens measurable

md5 `15de5298506fedada6f39f9015d934a2`. **All 160/160 tokens crashed.**
Not a tile-space property and not specific to coopmat or to the tsweep
variants — the driver segfaults on *any* buffer-storage compute dispatch:

| test | storage | kernel | result |
|---|---|---|---|
| default (coopmat path) | buffer | `linear_q4gsw_coopmat_*` | **segfault** |
| default | texture3d | `linear_q4gsw_tiled_*` | works, full numbers |
| `--baseline` (forced tiled) | buffer | `linear_q4gsw_tiled_*` | **segfault** |

Forcing the *tiled* kernel still crashes when storage is buffer, so storage —
not the shader — is the discriminator. Backtrace (`driverB-e0da99c1d1-crash-backtrace.txt`):

```
signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0 (read)
Cause: null pointer dereference
  #00 pc 0000000001e8d1bc  /vendor/lib64/hw/vulkan.samsung.so
  #01 pc 0000000001dd32f4  /vendor/lib64/hw/vulkan.samsung.so
  #02 vkcompute::vkapi::CommandBuffer::dispatch(...)
  #03 vkcompute::api::Context::register_shader_dispatch(...)
```
Null deref inside the driver's `vkCmdDispatch`, on the first dispatch.
`e0da99c1d1` is *"spal,pal_common: fix ResourceTracker not reaching HW via
decorators"* — a resource-tracking change. A decorator path that leaves a null
resource for buffer-backed descriptors matches this signature exactly.
**Consequence: no A/B is possible.** ExecuTorch's quantized coopmat path
requires buffer storage, so this driver cannot run it at all.

## Driver A (`f14c51b6f8`) — 4w

38/48 ranked. Excluded: 10 correctness-gate failures, 0 crashes, 0 fallbacks.

| rank | geomean GFLOP/s | % of best | token |
|---|---|---|---|
| 1 | 5367.0 | 100.0% | `tsweep_dbuf4_t128x128k16g22s32`  ← **production tile** |
| 2 | 5365.8 | 100.0% | `tsweep_dbuf4_t128x128k16g14s32` |
| 3 | 5082.9 | 94.7% | `tsweep_dbuf4_t64x128k16g41s32` |
| 4 | 4920.5 | 91.7% | `tsweep_dbuf4_t128x64k16g22s32` |
| 5 | 4843.2 | 90.2% | `tsweep_dbuf4_t128x128k32g42s32` |
| 6 | 4804.9 | 89.5% | `tsweep_dbuf4_t128x128k16g21s32` |
| 7 | 4803.7 | 89.5% | `tsweep_dbuf4_t64x128k16g22s32` |
| 8 | 4799.2 | 89.4% | `tsweep_dbuf4_t64x128k16g21s32` |
| 9 | 4771.5 | 88.9% | `tsweep_dbuf4_t64x128k16g12s64` |
| 10 | 4691.3 | 87.4% | `tsweep_dbuf4_t128x64k16g12s32` |

- spread best→worst **4.20x** (5367.0 → 1276.6), median 3391.7
- tokens within 1% of best: **2**
- the production *geometry* `t128x128k16g22s32` ranks **1 of 38 within dbuf4**.
  Note this is that geometry rebuilt as dbuf4; what ships for 4w is dbuf1. This
  says the geometry is the best dbuf4 geometry — **not** that dbuf4 matches or
  beats the shipped dbuf1 kernel, which was not measured.

## Driver A (`f14c51b6f8`) — 8da4w

50/112 ranked. Excluded: 59 correctness-gate failures + 3 driver segfaults.

| rank | geomean GFLOP/s | % of best | token |
|---|---|---|---|
| 1 | 6203.1 | 100.0% | `tsweep_dbuf4_t128x64k64g42s64` |
| 2 | 5699.7 | 91.9% | `tsweep_dbuf4_t128x64k64g18s64` |
| 3 | 5690.8 | 91.7% | `tsweep_dbuf4_t128x128k32g18s32` |
| 4 | 5251.0 | 84.7% | `tsweep_dbuf4_t64x64k32g24s32` |
| 5 | 5158.9 | 83.2% | `tsweep_dbuf4_t128x128k32g24s64` |
| 6 | 5147.5 | 83.0% | `tsweep_dbuf4_t128x32k32g24s32` |
| 7 | 5144.9 | 82.9% | `tsweep_dbuf4_t128x64k32g41s64` |
| 8 | 5003.3 | 80.7% | `tsweep_dbuf4_t128x128k64g24s64` |
| 9 | 4898.6 | 79.0% | `tsweep_dbuf4_t128x128k32g24s32` |
| 10 | 4859.2 | 78.3% | `tsweep_dbuf4_t128x64k32g24s32` |
| … | | | |
| 11 | 4788.1 | 77.2% | `tsweep_dbuf4_t64x32k32g12s64`  ← **production tile** |

- spread best→worst **2.65x** (6203.1 → 2343.9), median 4213.8
- tokens within 1% of best: **1**
- the production *geometry* `t64x32k32g12s64` ranks **11 of 50 within dbuf4**, at
  77.2% of the best dbuf4 geometry (i.e. 29.6% below it).
  ⚠️ **This is not 29.6% of available headroom over what ships.** What ships for
  8da4w is that geometry under **dbuf2** (`linear_dq8ca_qw_coopmat`, specs/027,
  shipped 07-12); here it was measured under **dbuf4**. No dbuf2 point was run,
  and memory `dbuf-variant-differs-by-quant-scheme` records dbuf2 as the 8da4w
  winner — so dbuf4's whole curve may sit below dbuf2. Establishing real headroom
  requires measuring the shipped dbuf2 kernel alongside these.

## Canary — driver A re-measured after flash→B→reflash→A

| scheme | token | before | after | drift |
|---|---|---|---|---|
| q4gsw | `tsweep_dbuf4_t128x128k16g22s32` | 5367.0 | 5369.3 | +0.04% |
| q4gsw | `tsweep_dbuf4_t128x64k16g12s32` | 4691.3 | 4695.7 | +0.09% |
| q4gsw | `tsweep_dbuf4_t128x64k16g22s32` | 4920.5 | 4922.1 | +0.03% |
| q4gsw | `tsweep_dbuf4_t64x128k16g41s32` | 5082.9 | 5082.5 | -0.01% |
| dq8ca | `tsweep_dbuf4_t128x128k32g18s32` | 5690.8 | 5692.0 | +0.02% |
| dq8ca | `tsweep_dbuf4_t128x64k64g18s64` | 5699.7 | 5701.4 | +0.03% |
| dq8ca | `tsweep_dbuf4_t128x64k64g42s64` | 6203.1 | 6206.0 | +0.05% |
| dq8ca | `tsweep_dbuf4_t64x32k32g12s64` | 4788.1 | 4787.0 | -0.02% |

Max |drift| **0.09%** across a full driver flash/reflash cycle. The rankings
above are not thermal or drift artifacts.

## Caveats

- **Microbench-only, not a production pick.** specs/026 vs specs/027 saw the
  8da4w microbench ranking nearly invert against e2e.
- Ranks dbuf4 tiles **against each other only**. No dbuf1/dbuf2 baseline was run,
  so nothing here says whether dbuf4 beats the shipped loop structure (dbuf1 for
  4w, dbuf2 for 8da4w). Every "% of best" is relative to the best *dbuf4* config.
- **The correctness gate is small-shape only.** `kCorrectnessShapes` tops out at
  K=4096 / M=128 and perf rows report `correctness = SKIPPED`, so a tile that is
  wrong specifically at prefill M=2048 would still be counted `ok` and ranked.
  Memory `rel13-dq8ca-prefill-wrong-m2048` is exactly that failure mode for dq8ca
  (wrong at 2048 while a short coherence check passes). Some of the 50 ranked
  8da4w tokens could be affected; a real-shape correctness case would close this.
- Primary M51; the existing specs/036 dbuf4 e2e summaries are the *secondary*
  board and a different metric. Not directly comparable.
