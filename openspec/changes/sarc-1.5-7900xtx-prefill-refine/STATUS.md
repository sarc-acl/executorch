# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-08 17:24 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. Nothing pushed.

## Running now
- Control workstation: native build of the parent (`build-both.sh 90fe4d013 parent`, build tag `parent`, then `parent-traced`), detached, niced.
- GPU host: nothing of this campaign (checked at start: only the idle `ollama serve`; no D-state workers; power state D0).

## Step 0 (device check), 2026-10-08 17:06 UTC
Passed: the GPU host (up 6 min after the owner's reboot) lists "Radeon RX 7900 XTX" (discrete, AMDVLK 2025.Q2.1 LLPC) with the
AMDVLK ICD; runtime PM active / D0; `pp_dpm_sclk` 500 / 0* / 2482 MHz; power cap 327 W; `power_dpm_force_performance_level` auto
(as found, not changed); hwmon index of the amdgpu is 3 this boot (the tools look it up by name). Other user logged in on the host.

## Done so far
1. Change directory, tools adapted for the two-machine workflow, thresholds fixed in `proposal.md` / `tools/thresholds.txt` (commit f1849af1a), before any measurement.
2. Tools smoke-tested on the GPU host without a GPU job: env.sh, others.sh (guard), sampler.py (measured interval 6.3 ms median, 7.1 ms max).
3. Submodules of the working copy fetched from their public GitHub URLs (read-only download, `git submodule update --init --recursive`; the clone had none), so that R5 exports can pin them: 30 of 30 populated.
4. `fused3sb` (M2a) dev-zone files from `origin/topic/rx7600-prefill-refine` committed, not selected by default (3179f8be3); read for shared writes (one writer per slot, execution barrier after every memory barrier, conditional barriers under a uniform `subgroupAny`).

## Notes / things to explain later
- The benchmark prompt is the kit's `prompt_2048.txt` ("the" x 2048, sha256 bfce65eb...), as the published contribution notes say; the GPU host also has a different `p2048tok.txt` (sha256 d1e7a8d7...), which is not used. CAMPAIGN.md section 4 names `p2048tok.txt` as the source of the published numbers; the contribution notes name the kit prompts. If the baseline is off by more than 3 % this is the first thing to check.
- `verify.sh` finds no unaligned prompt `r*.txt` in the stage directories (as in the sibling campaigns), so its "unaligned" lines are absent from the snapshot and from every candidate alike; the unaligned next-token comparison is `prompt_check.txt` (1972 tokens) in `e2e5.sh`.
- The fused kernels declare 32-lane subgroups; whether AMDVLK runs them as wave32 is open until the SDPA tiers run (risk, UNVERIFIED).

## Next
Parent build -> stage `aa` and `s0-parent-verify` -> push to the GPU host -> `q-aa.sh` (A/A, snapshot, parent SDPA tiers) -> calibrate -> baseline check against the published table.

## Decision needed from the owner
None yet.
