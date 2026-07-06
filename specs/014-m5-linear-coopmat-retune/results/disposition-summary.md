# Disposition Summary: M5 EVT1 `4w` Linear Coopmat Retune

Status as of 2026-07-05. Schema per `../data-model.md`'s Retuned Shader
Change / Documentation Clarification records.

## Shader changes (`linear_qw_coopmat.glsl`)

| name | risk_level | correctness_result | perf_result | disposition | disposition_reason |
|---|---|---|---|---|---|
| `fp16_accumulate` | precision_risk | not_yet_run | not_yet_run | pending | Blocked on M5 EVT1 hardware access (no device reachable this session — `adb devices` empty). See `tasks.md` Phase 5 (T011-T014). |
| `loop_flattening` | same_math_code_shape | not_yet_run | not_yet_run | pending | Blocked on M5 EVT1 hardware access. See `tasks.md` Phase 4 (T006-T010). |
| `vectorized_dequant` | same_math_code_shape | not_yet_run | not_yet_run | pending | Blocked on M5 EVT1 hardware access. See `tasks.md` Phase 4 (T006-T010). |

## Documentation clarification (`linear_dq8ca_qw_coopmat.glsl` / `QuantizedLinear.cpp`)

| files | finding_date | validation_gate | disposition |
|---|---|---|---|
| `linear_dq8ca_qw_coopmat.glsl`, `QuantizedLinear.cpp` (`add_linear_dqa_qw_node`) | 2026-06-30 | None — comment-only, no runtime effect | keep |

## Next steps

Phases 4 and 5 in `../tasks.md` are ready to run as soon as Samsung M5 EVT1
device access is available in a session (per
`.shared-context/instruction-for-ai/devices-and-access.md`). Neither phase
blocks the other; `fp16_accumulate`'s disposition does not gate
`loop_flattening`/`vectorized_dequant`'s, per this spec's Clarifications.
