# Research: M5 EVT1 8da4w T-tiled Baseline

## Decision 1: Export with the default (no `storage_override`) config -- this produces T-tiled, not a special "tiled mode"

**Decision**: Export each model's `8da4w` PTE at `ctx3072` using the
export config's default behavior -- i.e., omit
`backend.vulkan.storage_override` entirely, do not set it to `buffer`.

**Rationale**: Per constitution's Shader/Storage Configuration Taxonomy,
**T-tiled** is defined as "texture storage, tiled shader — the honest
baseline (what a user gets today)", produced by a plain texture `.pte`.
The `buffer`-override mechanism exists specifically to *enable* the
coopmat-eligible path (Buffer storage is a precondition for
`can_use_q4gsw_coopmat`/the `dq8ca` equivalent to ever fire) -- omitting
it is what makes this the honest, unmodified baseline, not an
implementation detail to get right by accident.

**Alternatives considered**: Setting `storage_override: buffer` and
relying on the eligibility gate naturally falling back to tiled for some
other reason -- rejected. That would produce a **B-tiled** measurement
(buffer storage, tiled shader -- the existing gotcha G9/constitution
taxonomy's "diagnostic baseline" tier), not T-tiled. The two are
deliberately different comparison points in this workstream's own
taxonomy; conflating them would produce a number that looks like a
baseline but isn't the one the report needs.

## Decision 2: Export at `ctx3072`, matching every other PTE in this workstream's Default Scope

**Decision**: `ctx3072` (i.e. `MAX_SEQ=MAX_CTX=3072`), not `ctx2304` (the
existing, stale 8B `8da4w` texture PTE's context length) or any other
value.

**Rationale**: Constitution's Default Scope for Every Benchmark fixes the
2048-prefill/1024-decode workload and requires `ctx3072` to serve it
comfortably. The existing `llama3_1_8b_8da4w_texture_ctx2304.pte` cannot
be reused for this feature's purpose even though it technically exists,
because `ctx2304` doesn't match the workload every other number in the
comparison table was measured at -- reusing it would produce a
speedup ratio comparing two different workload sizes, which is worse
than having no baseline at all (a wrong number that looks authoritative
vs. an honestly-missing one).

**Alternatives considered**: Reusing the stale `ctx2304` PTE for 8B to
save an export cycle -- rejected for the reason above.

## Decision 3: Sequence 1B -> 3B -> 8B, reuse this workstream's existing "why" verbatim

**Decision**: Same order and rationale as `specs/015` Decision 3 --
cheapest/fastest model first to prove the export+measure methodology,
most expensive/highest-watchdog-risk model last.

**Rationale**: No new reasoning needed; this feature is the same shape of
work (export + measure + report per model) that `specs/015` already
established a sequencing precedent for, on the same hardware, at the same
workload.

**Alternatives considered**: Doing 8B first since it's the most
report-impactful number -- rejected, matches `specs/015`'s own
established preference for proving methodology cheaply before spending
device time on the slowest, highest-risk model.

## Decision 4: Dispatch verification via ETDump kernel-name breakdown is sufficient here (no bind-time diagnostic needed)

**Decision**: Confirm each T-tiled run's dispatch via a standard ETDump
capture + `analyze_etdump_shaders.py --by kernel`, checking that the
linear kernel family shown is `linear_dq8ca_q4gsw_tiled_*`, not
`linear_dq8ca_q4gsw_coopmat_*`. Do not additionally require the
`ET_VK_DEBUG_ENCODE_DISPATCH` bind-time diagnostic (built and used this
session for G6/Q11) unless the ETDump result is ambiguous or surprising.

**Rationale**: Gotcha G6's known failure mode is specifically ETDump
under-reporting coopmat as tiled (a genuinely-coopmat dispatch showing up
mislabeled as `_tiled`) -- there is no known or hypothesized failure mode
in the *other* direction (a genuinely-tiled dispatch showing up
mislabeled as `_coopmat`), and structurally, the coopmat shaders require
Buffer storage while this feature's PTEs are texture storage by
construction (Decision 1) -- the coopmat `ShaderInfo` for this op family
isn't even a candidate the eligibility gate would consider. The stronger,
multi-method verification bar `specs/015` ultimately needed (Decision 8)
was necessary because the *positive* coopmat claim needed defending
against a demonstrated attribution bug; a T-tiled baseline's claim is the
negative case that bug doesn't threaten. If a run somehow *does* show a
coopmat kernel name, that's the surprising result that would warrant
escalating to the stronger method, not something to design in as
required up front.

**Alternatives considered**: Requiring the full bind-time diagnostic for
every run regardless -- rejected as disproportionate verification effort
for a claim this workstream's own gotcha doesn't actually put at risk;
would burn device time re-deriving a verification bar this feature
doesn't need.

## Decision 5: 3-run mean + CoV, identical convention to every other number in `specs/015`

**Decision**: Same as `specs/015` Decision 5 -- 3 repeated timed runs per
model, reporting mean and CoV, with clock-pin verification done once per
session (not per-run) via the existing GFLOP/s cross-check.

**Rationale**: This is the exact convention the `4w` T-tiled baseline
this feature's numbers will sit alongside was itself measured with
(`RESULTS-SUMMARY.md`'s trusted anchor) -- matching it is what makes the
resulting ratio comparable, not a separate methodological choice this
feature is free to make on its own.

**Alternatives considered**: A single run per model (faster, less device
time) -- rejected, would produce a baseline with a lower evidentiary bar
than the number it's being compared against, undermining the ratio's own
credibility.
