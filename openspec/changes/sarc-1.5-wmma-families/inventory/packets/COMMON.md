# Company-side handoff: bounded matrix preparation

For the owner to relay. This document does not start jobs or contact a company host. It specifies inputs
and evidence for separately launched device stages after independent inventory review and verified
prerequisites. Source base is `6178971845`; use the reviewed manifests from this inventory commit.
RX 7600 and XTX results may be public; M51 numbers and raw evidence remain private.

1. Inspect existing status, processes, queues, locks, artifacts and HOLD before starting. Stay on the assigned
   branch, checkout and unit. Respect the task's cap and reviewed finite candidate list. No attention tuning,
   decode/fp-linear tuning, parameter search, new bodies, upstream baseline, promotion, push or PR is implied.
2. Record unit and complete driver/software identity privately. Query vendor/driver ID, cooperative matrices
   with shapes/types/scope, subgroup defaults/ranges/control/computeFullSubgroups, workgroup and shared-memory
   limits, and required shader/storage features. Compare every manifest entry using `LEGALITY.md` and fill
   its target legality status with evidence. Illegal variants are pruned before GPU launches; record why.
   Current manifest target statuses are all pending, including their originating device.
3. Validate all six models: 1B/3B/8B x 4w/8da4w, context length 2560, quantization group size and embedding
   configuration against `openspec/changes/sarc-1.5-llamacpp-compare/kit/MODELS.md`. Read context length through the runtime, not just the share manifest.
   MODELS.md labels group size 128 and embeddings (-E 4,32) as inferred, not runtime
   metadata; substantiate them using export provenance or direct program/tensor inspection,
   and retain an explicit unknown if unavailable. Do not turn an inference into verification. Record local digests/sizes and tokenizer/prompt identity. Do not
   re-export or copy model files to public source. Export differences stop dependent work for the owner.
4. Before measurement pre-register repeat count, clock-floor rule, thermal-reason mask, cooldown/plateau
   rule, sampling cadence/minimum samples, busy ceiling and baseline comparison tolerance. Use parent
   snapshot and A/A, immutable recursive exported-commit builds and unmodified correctness gate on the exact
   binaries/environment timed. Record compiler identity and shipped-golden outcome, including inherited
   differences; historical exceptions are not silently extended. No build during timed sessions.
5. Re-measure this device's final linear configuration in the same session as each legal candidate set.
   Keep attention configuration and other environment identical across comparison arms. If the full profile
   also changes attention, adapt a dev-zone linear-only registration/control in the assigned stage; a
   full-profile swap is not a linear comparison. Record all dispatched linear names and attention controls.
   Changing int8 layout requires rebuilding the graph consistently, not swapping shaders over packed data.
6. R6: `llama_main`, unchanged 2048-token prompt, max_new_tokens=1, temperature=0, warmup, fresh process,
   page-cache fill per model for both arms, interleaving, median of five valid runs unless A/A preregisters
   more. Lock one job per unit; sample clocks, thermal reasons, temperature/power and foreign workload during
   the run, not only before/after. Require rc=0, 2048 prompt/0 generated tokens and calibrated sensor rules.
   Preserve invalid rows and replace them; diagnose a failed gate before retrying. Do not widen thresholds.
7. Run the unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, including its decode checks;
   compare to the parent snapshot with rates removed and intended dispatch differences explicit. Include
   next-token checks on timed, real-text and unaligned prompts, arithmetic/reference-error or bit-identity
   evidence as appropriate, and shader synchronization review before gating. Attention remains controlled;
   do not launch an attention experiment because the historical template mentions one.
8. Return raw rows/logs and export/build/binary/environment manifests under the artifact root. Each row
   identifies unit, driver, commit, binary, profile, candidate, session, cell, validity/reason, clock/sensor and
   foreign-work evidence. Return small publishable CSVs and gate summaries under the device change directory.
   Preserve rejected/invalid/interrupted runs and the planned-vs-completed list. Final acceptance belongs to
   family selection: <=3% six-cell geomean slowdown and <=8% per-cell slowdown against tuned linear control,
   with uncertainty reported. No winner or minimal family set is inferred in this inventory.

Device loss stops the queue without retry/reset/reboot/driver reload. Foreign work means wait and report;
never kill others' processes or change services, host policy, clocks or power. Every detached queue obeys
coordinator HOLD at job boundaries. Record an owner question once, finish independent work, then wait;
check for an answer at most once per 20 minutes. Stop/restart/resume decisions remain with the owner.

Review evidence must be available to an independent reviewer under the privacy rules. No direct messages
or host access from this inventory actor are authorized. Internal paths, names, serials and IPs stay private.
