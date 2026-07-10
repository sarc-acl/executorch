# Data Model: Workstream Agent Housekeeping

## Gotcha Entry

One consolidated, citable lesson in `.specify/memory/gotchas.md`.

| Field | Type | Notes |
|---|---|---|
| `id` | string | `G<N>`, stable once assigned, never reused even if an entry is later marked resolved (Decision 2) |
| `title` | string | One-line symptom, e.g. "Android `install` can silently fail on an unrelated target, staling `libvulkan_backend.a`" |
| `symptom` | text | What an agent would actually observe (error text, wrong output, wasted time) |
| `root_cause` | text | The actual mechanism, in enough detail to recognize a recurrence |
| `fix_or_workaround` | text | The concrete action that resolves or avoids it |
| `citation` | string | `specs/NNN-.../research.md` (or this feature's own spec, for entries with no prior spec home) pointing to the full original narrative |
| `status` | enum | `open` (workaround only, underlying issue not fixed) / `resolved as of <ref>` (the underlying issue itself was fixed -- entry kept for history, per spec Edge Cases) |

**Initial 10 entries (G1-G10), per spec FR-003, in the order listed there**:

| id | title | citation |
|---|---|---|
| G1 | Android `install` can fail on an unrelated target (`executor_runner`, stale host-arch `flatccrt.a`), silently leaving `libvulkan_backend.a` un-updated for any downstream sub-build | This session's own build narrative (no prior spec; first documented here) |
| G2 | `ET_VK_FORCE_BUFFER` does not exist in this repo; real mechanism is `backend.vulkan.storage_override: buffer` | `specs/015-m5-e2e-wmma-validation/research.md` Decision 6 |
| G3 | Non-editable `.venv` silently no-ops AOT/export Python code changes | `specs/015-m5-e2e-wmma-validation/research.md` Decision 6 |
| G4 | Exported `.pte` files must land directly in `/local/yanwen.xu/workspace/.pte_out`, never `/tmp`/scratch | Constitution v2.3.0 (already a principle-level rule; cross-referenced, not duplicated) |
| G5 | `/tmp` is small (20GB) and this sandbox denies `rm -rf` even on one's own scratch files -- use `mv`, not `rm` | This session's own narrative (no prior spec) |
| G6 | ETDump's per-event `kernel_name` is not reliable dispatch evidence in the full LLaMA graph context | `specs/015-m5-e2e-wmma-validation/research.md` Decision 7 (and its reversal) |
| G7 | Two similarly-named SDPA benchmark harnesses exist; only `test_sdpa_coopmat_bench.cpp` is correct | `specs/016-m5-linear-sdpa-microbench/spec.md` Clarifications |
| G8 | Don't conclude a CMake target "isn't wired in" from a prefiltered grep -- grep the raw file directly | `specs/016-m5-linear-sdpa-microbench/tasks.md` T005 |
| G9 | The production linear-coopmat shader retune (commit `133044739`) was committed pre-hardware-validation; the MiniPC/tile-sweep comparison baseline used a different dispatch mechanism than this repo's production path | `specs/015-m5-e2e-wmma-validation/research.md` Decision 7's "post-completion lead" |
| G10 | M5 EVT1 is a shared device -- confirm with the user before assuming it's free | This session's own narrative (no prior spec); also see project memory `project-m5-device-sharing` |

The header (see Lifecycle below) also flags one standing risk: `CLAUDE.md`
could be silently overwritten by a future `install_executorch.sh` re-sync
that regenerates the stock upstream template, erasing the pointer block
User Story 1 adds -- out of scope for this feature to prevent (Edge Cases),
but noted here so a future session notices immediately if it recurs.

## `CLAUDE.md` Pointer Block

The new lines added near the top of `./CLAUDE.md`. Fields (conceptual, not
literal markdown fields -- this is prose, not a table in the actual file):
which document governs (`constitution.md`), the active target (M5 EVT1),
where shared tooling lives (`.shared-context/`), and the one-sentence
caveat that a `.shared-context/instruction-for-ai/` doc's literal
mechanism can be wrong for this repo (pointing to `gotchas.md` for
specifics, per G2).

## Constitution Cross-Reference

Two small additions inside existing principles (not new principles):

| Location | Addition |
|---|---|
| Principle VI (Verify With Tools, Never Assume) | One example sentence citing G6, pointing to `gotchas.md` |
| Principle X (Consult `instruction-for-ai` Before Acting) | One caveat sentence citing G2 as the concrete instance of "a cited mechanism can be wrong for this repo", pointing to `gotchas.md` |
| Development Workflow (new subsection or existing "Issue & Open-Question Tracking") | One paragraph introducing `gotchas.md`, its append convention, and its relationship to `open-questions.md` (parallel, not a replacement) |

## Lifecycle

```
Gotcha Entry created (status=open) --(root-cause investigation completes AND the
  underlying code/process issue is actually fixed, not just worked around)-->
  status = "resolved as of <spec/commit ref>"
  [entry text stays, historical record per spec Edge Cases -- never deleted]

Future session hits a new multi-hour or repeat-mistake operational issue -->
  append a new G<N+1> entry, same schema, per FR-004a's header instruction
  [id numbers are never reused, even for resolved/removed entries]
```
