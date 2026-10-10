# Status of the llama.cpp comparison on the company side's devices (COMPANY-SIDE.md section 10)

Updated 2026-10-10 02:30 UTC. Nothing is pushed. The three devices run in parallel (owner decision 2026-10-10 01:10 UTC).

| device | 10.1 state | where |
|---|---|---|
| RX 7900 XTX | session `rx7900xtx-1` done (AMDVLK 2025.Q2.1, all 14 arms, three models, both timers, both quantizations); ARMS.md committed before it; rows in `cells.csv`. Optional extras (decode, real-text prompt) not done yet | `7900xtx/` |
| RX 7600 | round 2 final as tuned arm: settings screen done, ARMS.md committed, timed session `rx7600-r2` running (RADV Mesa 26.2.3); the 2026-10-08 files are kept in `rx7600/history-2026-10-08/` | `rx7600/` |
| M51 | llama.cpp b11430 builds for Android and starts on the board's Vulkan driver; settings screen done locally; the timed session has not started. Every number stays local; only the relative statement will be committed (`m51/README.md`) | local only |

10.2 (HIP) is not part of this task (owner decision 2026-10-09).

## RX 7900 XTX in one paragraph

Tuned `4w` is ahead of llama.cpp Vulkan Q4_0 and Q4_K_M at every model size, by either timer and at both settings; `lc` pays the
card's clock ramp (its `clock_low` runs are counted and shown); stock, parent and tuned were measured in the same session, so the
review's cross-session objection no longer applies. The details, deviations and audit are in `7900xtx/README.md`.
