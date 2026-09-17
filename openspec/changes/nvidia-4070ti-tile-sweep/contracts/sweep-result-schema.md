# specs/036 sweep result schema (036.1)

Extends specs/028's `e2e-ranking-schema.md` field vocabulary. The M5-specific
provenance fields (`driver_hash`, `board`, `clocks_pinned`) are replaced by a
portable fingerprint block; everything measurement-shaped keeps 028's names
(`candidate_token`, `prefill_tok_s`, `improvement_pct`, ...).

Three committed artifacts per (device, shader), all under `results/`:

## 1. `runs_<device_slug>_<shader>.jsonl` — every rep, append-only

Common fields on every record: `ts`, `schema_version` ("036.1"),
`device_slug`, `shader` ("q4gsw" | "dq8ca"), `baseline_id` (increments each
re-baseline after unrecovered drift), `stage`.

Per `stage`:

- `fingerprint` (first record of a file): the full device_fingerprint block —
  `device_name`, `device_type`, `driver_id`, `driver_info`, `driver_version`,
  `api_version`, `subgroup_size_default`, `min/max_subgroup_size`,
  `max_compute_shared_memory_size`, `max_compute_work_group_invocations`,
  `os`, `perf_level` ("unknown" allowed), `git_sha`, `captured_at`.
- `baseline` / `control`: `candidate_token: "CONTROL"`, `rep`,
  `prefill_tok_s`, `model_load_ms`, `output_hash`.
- `baseline_summary`: `median_prefill_tok_s`, `noise_floor_cov`,
  `control_hash`.
- `gate`: `candidate_token`, `gate_status`
  (`pass` | `correctness_fail` | `alignment_fallback` | `missing_shader`),
  `gate_fails`, `gate_dispatched`.
- `screen` / `confirm` / `validate`: `candidate_token`, `rep`,
  `prefill_tok_s`, `model_load_ms`, `output_hash`, `output_match`
  (false = silent miscompute — the record is kept but the token is excluded
  from ranking and blocklisted).
- `drift` / `rebaseline`: drift-ladder bookkeeping (`attempt`, `observed` /
  `reason`).

## 2. `optuna_journal_<device_slug>_<shader>.log` — search state

Optuna JournalStorage (JSON lines). This is the resume state: rerunning the
same `sweep.py` command continues the study. Trial user-attr `token` carries
the tsweep token; PRUNED trials are ask-time rejections or deterministic
failures (reasons live in the blocklist, not the journal).

## 3. `sweep_summary_<device_slug>_<shader>.json` — the answer

`schema_version`, `shader`, `device_slug`, `fingerprint`, `group_size`,
`budget`, `measured_trials`, `pruned` (counter by reason), `blocklist_size`,
`baseline_median_tok_s`, `noise_floor_cov`, `finalists` (list of
`{token, screen_tok_s, confirm_median_tok_s, improvement_pct}` sorted by
confirmed median), `winner` (first finalist), `remeasure_pending` (tokens
measured inside a drifted window — re-screen these before trusting rank),
`ts`.

## Auxiliary (committed)

- `blocklist_<device_slug>_<shader>.jsonl`: `{token, reason, detail, ts}`,
  reasons: `gate_correctness_fail`, `gate_alignment_fallback`,
  `gate_missing_shader`, `glslc_failure`, `output_miscompute`. Tokens here
  are never re-proposed on this device.
- `microbench_780m_<shader>.jsonl` + `rank-correlation-780m-<shader>.md`:
  component C artifacts (see protocol step 3).
- `replay-780m/`: the specs/035 round-1 raw TSVs (`{4w,8da4w}_{gate,e2e}.tsv`
  + `original_driver.sh`), kept both as provenance and as the `--dry-run`
  replay fixture.
