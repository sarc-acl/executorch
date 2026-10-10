# Radeon RX 7600: arms of the comparison, round 2 final as the tuned arm (written before the timed session, 2026-10-10)

This replaces the 2026-10-08 arms (round 1 tuned arm; kept as history in `history-2026-10-08/`). Device: Radeon RX 7600
(RADV NAVI33, 8 GB), the workstation's own card. **Driver for every arm, ExecuTorch and llama.cpp alike: user-space RADV Mesa
26.2.3 (`31e9a6b2e9`)**, Vulkan 1.4.354, through `VK_ICD_FILENAMES`; no other driver is used. The tuning campaign
(`sarc-1.5-rx7600-prefill-refine`, round 2) is finished. llama.cpp reports `int dot: 1` and `matrix cores: KHR_coopmat` on this device.

Validity: the campaign's rule as `row.py` applies it (median clock in the prefill window at least 2420 MHz, the campaign's
calibrated floor; at least five clock samples; exactly 2048 prompt tokens; no other GPU user), sampled every 10 ms from the
card's hwmon by `kit/hosts/rx7600/host.sh` (unchanged since the 2026-10-08 session). The card has no per-client engine
accounting, so there is no foreign-busy ceiling. Host builds of anyone are recorded per run (`<run>.builds`) and a
one-second process watcher runs beside the session; builds on the workstation are allowed during a session (owner decision
2026-10-09) and are listed in the README, they do not invalidate a run.

## ExecuTorch Vulkan (each with `4w` and `8da4w`)

| arm | build | selection (`env` of the arm, besides the ICD) |
|---|---|---|
| `stock` | upstream `release/1.5` at `985c1ceccc` plus the compile-only backport (`stock-backport-03f41d2031.patch` of `sarc-1.5-e2e-benchmark`), built natively as in the 2026-10-08 session (podman does not run on this host, so `build-stock.sh` was not used); the same binary as then | none |
| `sarc` | `f5f1bf10c`, the campaign's parent (`dev/1.5` at the start of the campaign), the campaign's own binary of its round 2 final session `r2-final`; the same binary as in the 2026-10-08 session | `ET_VK_SARC_UNVERIFIED=1` |
| `tuned` | `73648f5bd`, the campaign's round 2 final build (`f2`): an unmerged development branch, the campaign's own binary of its final session `r2-final` | `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine5` (the campaign's round 2 final configuration) |

sha256 of `llama_main`: stock `5869094d…`, sarc `cfb5fb8c…`, tuned `b062ca10…` (sarc and tuned identical to the campaign's
`r2-final`). The 2026-10-08 tuned arm was `18cc0d53a` with profile `rx7600-refine2` (`ca2d39be…`).

`.pte` files: the campaign's exports, `*_embq_ctx3072.pte` (context 3072, 4-bit embeddings, group 128), not the 2560-context
shared exports of `MODELS.md`. Sizes: 1B 789,135,872 / 827,832,192, 3B 1,886,556,160 / 1,987,087,360, 8B 4,173,751,424 /
4,408,429,824 bytes (`4w` / `8da4w`). llama.cpp runs with `-c 2560` as the kit fixes it.

## llama.cpp `b11430`, Vulkan, built on the host with `-DGGML_VULKAN=ON -DGGML_NATIVE=ON`

`q4_0` and `q4_k_m`, each as `default` and `best`, by both timers (`lc`, `lb`): 8 arms. Exact lines: `arms.tsv`
(`llama-completion` and `llama-bench` are the `b11430` binaries; the same binaries and GGUF files as on 2026-10-08, see
`history-2026-10-08/README.md` for how the GGUF files were made).

## Settings screen (fixes `best`)

`screen.csv`, session `screen1`, 2026-10-10 01:08 to 01:34 UTC, on 1B and 3B, both quantizations: `llama-bench` (`lb`, one
process of five repetitions each) at the default setting (`-ub 512`, flash attention auto) and at `-b 2048` with `-ub {512,
1024, 2048}` x flash attention {on, off}; `llama-completion` default (`lc`, three runs); the tuned `4w` arm as an anchor.
Result (`lb` tok/s, Q4_0 then Q4_K_M; 1B / 3B): default 5853 / 2450 and 5013 / 2042; `-ub 512` on 5845 / 2450 and 5009 / 2041,
off 4245 / 459 and 3795 / 443; `-ub 1024` on 5908 / 2284 and 4940 / 1917, off 4285 / 460 and 3764 / 442; `-ub 2048` on
5797 / 301 and 4954 / 136, off 3950 / 180 and 3536 / 104.

Reading: on 1B every flash-attention-on setting is within 2 % of the others; on 3B flash attention off, and any prompt
batch above 1024 tokens, falls to a slow path (a factor of 5 to 20 below flash attention on at `-ub 512`) and `-ub 1024` on is 6 to 7 %
slower than `-ub 512` on. By the rule of the screen (highest geometric mean of the two models' `lb` medians, per
quantization) the default and `-ub 512` on tie within 0.1 % (3787 and 3784 for Q4_0; 3199 and 3197 for Q4_K_M): no setting is better than
the default beyond noise. `best` is therefore the highest-ranked setting other than the default, `-b 2048 -ub 512 -fa on`, for both
quantizations (it differs from the 2026-10-08 `best`, `-ub 1024`, which this screen shows to be 6 to 7 % slower on 3B). `best` and
`default` are both measured, so the table shows both. The cause of the slow path is UNVERIFIED (as on 2026-10-08). The kit's list was
tried in full; no other llama.cpp option was screened.

A configure step of another job on the workstation (single `cmake` process, 01:20:07 to 01:20:50 UTC, seen by the process
watcher) overlapped two 3B screen runs (`-ub 1024` on and off); their values agree with the settings around them and the
screen does not depend on them.
