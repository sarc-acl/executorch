# Research: MiniPC RDNA3 Handoff Report

## Decision 1: Headline findings, verified against each source report

**Finding**: Read every prior feature's results file directly (not
recalled from memory) to source the consolidated report's numbers:

| Spec | Headline |
|---|---|
| `001` | Baseline e2e/microbench numbers established, coopmat/WMMA excluded (default behavior) -- the comparison floor for everything after |
| `002` | ETDump-based dispatch-confirmation methodology established (tier-1 tooling this whole workstream reuses) |
| `003` | WMMA candidates identified: prefill linear GEMM (blocked by a rank-3 output + `TEXTURE_3D` storage bug, not missing code), SDPA (no WMMA implementation existed at the time), decode linear GEMV (no WMMA-capable kernel exists, structural) |
| `004` | Buffer vs. `TEXTURE_3D` storage is "effectively free" for the large majority of cases (46/48 prefill, 35/48 decode) |
| `005` | E2E speedup target set (no measured result -- a goal-setting feature) |
| `006` | `004`'s finding confirmed at e2e scale, once cross-session prefill variance (a real, hardware-level property, not a storage effect) is controlled for |
| `007` | WMMA microbench, `4w`: **+60.6% faster**; `8da4w`: **-15.2% slower** (regression) |
| `008` | `8da4w` parameter sweep: best found config (`config 5`) averages **+18.2%** vs. shipped across representative shapes; a real shader bug found and fixed along the way |
| `009` | E2E tok/s, `4w`: **+77.8% faster** (consistent w/ `007`); `8da4w`: **-3.2% slower** (consistent direction, smaller e2e magnitude); found and fixed the `force_fp16`/storage-override conflict bug that had silently defeated `Buffer` storage workstream-wide until this feature |
| `010` | SDPA coopmat microbench (tier-1): **+66.8% average**, all 3 models real-effect |
| `011` | SDPA coopmat e2e (tier-2): **+27.3% average** across 6/6 configurations, all consistent in direction with `010` |
| `012` | Decode WMMA feasibility: roofline analysis finds decode's linear GEMV kernel **memory-bandwidth-bound by a 12-50x margin** -- not worth building a WMMA shader for; two alternative directions named (more aggressive quantization, or batching/speculative decoding for a real `M>1`) |

**Decision**: Cite every number above verbatim in the consolidated report,
each tagged with its source spec and tier (microbenchmark vs. e2e).

**Alternatives considered**: Re-deriving summary statistics from raw logs.
Rejected -- every one of these numbers is already a validated, published
figure in its own feature's results file; re-deriving would only risk
introducing a transcription error a prior feature already avoided.

## Decision 2: Open items to name explicitly

**Decision**: Per FR-003, the report must explicitly name:
- The deferred `8da4w` production-gating decision (`009`'s regression is
  real; no decision yet on whether to gate coopmat off for `8da4w` in
  production -- explicitly deferred by the user, owner has no plan yet).
- Decode SDPA's two GEMV kernels -- `012` covered only decode's linear
  GEMV; decode SDPA is expected to share the same `M=1`/bandwidth-bound
  conclusion (same structural property) but was never itself measured.
- The two directions `012` named but did not pursue: more aggressive
  weight quantization (a separate quantization-research effort, not a
  shader change), and batching/speculative decoding (batching doesn't fit
  the on-device single-user target at all; speculative decoding could
  apply but is a model-serving/algorithm feature orthogonal to this
  workstream's Vulkan-backend scope).
- Samsung/Xclipse validation itself -- every finding above is `rocky-ryzen`
  MiniPC-only; none of it has been confirmed on the actual target device
  (constitution Principle II).

**Alternatives considered**: None -- these are already-established facts
from this session, not new research.

## Decision 3: Repo state, read directly

**Finding**: `git branch --show-current` → `quant-perf-optimization`.
`git log --oneline -5` shows the last actual commit is
`d8800fb02e` ("Amend constitution: add Verify With Tools, Never Assume").
Every spec `007` onward, every production-code fix made during this
session (`QuantizedLinear.cpp`'s rank-3 guard, `tag_memory_meta_pass.py`'s
`force_fp16`/storage fix, the SDPA coopmat import and its
`GemmCoopmat.cpp`/`.h` workaround restoration, all new test files) is
**uncommitted** -- 71 files changed (`git status --short`: 47 added, 11
modified, 3 modified-and-staged, 8 untracked).

**Decision**: State this plainly in the report (FR-004): a clone of the
current remote branch today would contain none of specs `007`-`013` or
their underlying code fixes. Per Clarifications, committing/pushing is
named as the explicit prerequisite, not performed by this feature.

**Alternatives considered**: None -- this is a factual state, not a
decision this feature makes.

## Decision 4: `.pte` exports and report specificity -- per Clarifications

**Decision**: The report does not detail export mechanics or arrange file
transfer for the (gitignored) `.pte` files -- the user re-exports
independently on the Samsung/Xclipse machine, likely with a different
export configuration anyway. The Samsung/Xclipse runbook section is
high-level guidance and pointers to this workstream's existing
`quickstart.md` files, not an exhaustive, untested command script.

**Alternatives considered**: None -- directly resolved by the user's own
Clarifications answers; no further research needed.
