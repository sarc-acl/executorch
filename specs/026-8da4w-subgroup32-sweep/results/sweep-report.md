# 8da4w subgroup32-Reopen Sweep: Report

**Feature**: `specs/026-8da4w-subgroup32-sweep` | **Date**: 2026-07-11 | **Target**: M5 EVT1
(`xgpusw-debug08`, driver `f14c51b6f8` / `c9861e9906d03fa2c7d48b804e1a1c80`, clocks pinned
509/2730/663 MHz)

## Result at a glance

**`axis_disposition: subgroup32_wins`.** `SUBGROUP_SIZE=32` is legal (no compiler crash, at
any of 5 tested tile shapes) and, at one specific tile shape (`64×64/K16/2×1`), is both fully
correct across all 10 representative shapes *and* faster than `025`'s standing subgroup=64
winner: **2207.2 GFLOP/s vs 1736.0 GFLOP/s — a 27.1% improvement** (3-run mean, CoV 0.13%).
This reverses `025`'s exclusion-by-assumption: the axis should be open, not closed — but at a
*different* tile shape than the ones two prior single-shape probes (`025`'s T014, and this
session's earlier `sg32test` check) happened to test, both of which turn out to be exactly
the shapes where subgroup=32 is broadly *incorrect*.

## Scope note (deviation from tasks.md's full staged-search plan)

This implementation pass used a hand-selected 5-shape spread (`16×16/K16/1×1`,
`64×64/K16/2×1`, `128×32/K16/1×2` — `025`'s winner shape, `128×64/K32/2×2` — the shipped
shape, `64×128/K16/4×1`) rather than a full `enumerate_configs.py` → `score_and_shortlist.py`
run over the entire re-derived legal space (est. ~1000+ candidates). A clear, large-margin
winner (27% over the standing best, with a stark correctness split by tile shape) emerged
from this spread well within budget (2 candidates measured on hardware vs. a 30-measurement
cap) — consistent with spec SC-005's "target: far fewer" framing. The full enumeration
scripts (`tile_constraints.py`, T003) were still implemented per plan, so a follow-up can run
the complete search if a more exhaustive answer is wanted; this report's conclusion is
sufficient to answer the feature's core question (does *any* subgroup=32 config beat `025`'s
winner) but does not claim to have found the *global* optimum across the full space.

## User Story 1: SUBGROUP_SIZE=32 legality across 5 tile shapes

**No crash at any tested shape.** See `results/legality-summary.md` /
`subgroup32_legality.json`. `025`'s T014 and this session's earlier `sg32test` probe (both
one shape) are confirmed, not contradicted — generalized to a proper 5-shape spread.

## User Story 2: Correctness matrix (full 10-shape set, per candidate)

| Candidate | All shapes pass? | Failing shapes |
|---|---|---|
| `t16x16k16g11s32` | ✅ yes | — |
| `t64x64k16g21s32` | ✅ yes | — |
| `t128x32k16g12s32` (`025`'s winner shape) | ❌ no | 8/10 shapes |
| `t128x64k32g22s32` (shipped shape) | ❌ no | 8/10 shapes |
| `t64x128k16g41s32` | ❌ no | 1/10 (`M256_K128_N64`) |

**Key finding**: correctness at `SUBGROUP_SIZE=32` is sharply tile-shape-dependent — not a
uniform property of the axis. The two shapes prior probes happened to test
(`128×32/K16/1×2` and `128×64/K32/2×2`) are exactly the two *worst* shapes in this spread
(8/10 failures each); the smaller/differently-gridded shapes (`16×16`, `64×64`) are fully
correct. This is the concrete mechanism behind why two prior single-shape checks reached an
incomplete picture. Full per-shape data: `correctness_matrix.json`.

## User Story 3: Performance search and winner

Only the 2 fully-correct candidates were eligible for performance measurement (spec FR-004);
the 3 shape-dependently-incorrect candidates were excluded from ranking, not measured for
performance (`round3_results.json` `elimination_reason` fields).

| Candidate | Mean GFLOP/s (M=2048) | vs. `025` winner (1736.0) |
|---|---|---|
| **`t64x64k16g21s32`** | **2207.2** (3-run, CoV 0.13%) | **+27.1%** |
| `t16x16k16g11s32` | 1031.0 (1-run) | -40.6% |

**Winner: `t64x64k16g21s32`** — `WG_TILE_M=64, WG_TILE_N=64, WG_TILE_K=16, SG_GRID_X=2,
SG_GRID_Y=1, SUBGROUP_SIZE=32`, `dbuf2` loop structure (unchanged from `025`).

- `subgroup_size_used`: **32**.
- `comparison_vs_025_winner`: **1.271x** (2207.2 / 1736.0).
- `comparison_vs_shipped_8da4w` (pre-`025`, 1688.1 GFLOP/s): **1.307x**.
- `comparison_vs_4w_winner` (`022`, 2518.77 GFLOP/s, different shader/precision, informational
  only): 0.877x — still short of `4w`, consistent with `specs/024`'s premise that `8da4w`
  underperforms `4w` on this hardware; this feature narrows that gap without closing it.
- `spirv_verified`: **true** — 8 `OpCooperativeMatrixMulAddKHR` sites, `OpTypeCooperativeMatrixKHR`
  with `%char` (int8) component type, `REQUIRED_SUBGROUP_SIZE = 32` embedded correctly in the
  generated GLSL header (confirmed not a mislabeled fallback, Constitution Principle VI).
- `tie_broken`: false (clear margin, no tie).

## Probe disposition (spec FR-012/SC-007)

The session's ad-hoc `sg32test` shader/binding (shipped-shape-only probe) is **superseded**
by this feature's `tsweep_t128x64k32g22s32` canonical entry, which covers the identical
shape/tile combination and produced the identical correctness verdict (8/10 shapes fail).
The literal `"sg32test"` allow-list entry in `QuantizedLinear.cpp`'s `dq8ca_coopmat_variant()`
has been removed; the `sg32test` yaml `shader_variants` entry is retained only as a
superseded/documented historical artifact pending a follow-up cleanup commit (see
`tasks.md` T036).

## Shader comment update (Constitution Principle V)

See `results/shader-comment-update.diff` — proposed replacement for
`linear_dq8ca_qw_coopmat.glsl`/`.yaml`'s header comment, reflecting this feature's actual
shape-broad evidence instead of the stale blanket-crash claim.

## Recommendation

`recommendation: productionize_candidate` — `t64x64k16g21s32` (subgroup=32) is a real,
statistically confident (+27.1%, CoV 0.13%), correctness-verified (all 10 representative
shapes), genuinely-dispatching (SPIR-V confirmed) improvement over `025`'s standing winner.
Per this workstream's existing Tier-1/Tier-2 convention (`025`'s own precedent), this is
Tier-1 (shader microbenchmark) evidence only — a Tier-2 (`.pte` end-to-end tok/s) validation
is recommended before shipping this as the new default `8da4w` configuration, and actually
applying the shader-comment diff / promoting this token to production dispatch is a separate
follow-on decision, not made by this feature itself (spec Assumptions).

## Search cost (SC-005/SC-006)

5 candidates probed for legality (not counted against budget, per data-model.md), 5 for
correctness (not counted against budget), 2 taken to real performance measurement on
hardware — **2/30 = 6.7%** of the budget cap, and a small fraction of the full re-derived
legal space (exact count not computed in this pass — see Scope Note).

## Pruning audit

Every candidate's fate is traceable without re-running anything: `subgroup32_legality.json`
(compile status), `correctness_matrix.json` (per-shape pass/fail + failing shapes named),
`round3_results.json` (`elimination_reason` naming the specific failing shapes or the
performance cutoff).
