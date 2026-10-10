# Status of the llama.cpp comparison on the company side's devices (COMPANY-SIDE.md section 10)

Updated 2026-10-10 07:10 UTC. Nothing is pushed. The three devices run in parallel (owner decision 2026-10-10 01:10 UTC).

| device | 10.1 state | where |
|---|---|---|
| RX 7900 XTX | done: session `rx7900xtx-1` (AMDVLK 2025.Q2.1, all 14 arms, three models, both timers, both quantizations); ARMS.md committed before it; rows in `cells.csv`; extras (real-text prompt variant, decode with and without the final configuration) done | `7900xtx/` |
| RX 7600 | done: round 2 final as tuned arm, session `rx7600-r2` (RADV Mesa 26.2.3, all 14 arms, three models); ARMS.md committed before it; rows replaced in `cells.csv`; the 2026-10-08 files are kept in `rx7600/history-2026-10-08/` | `rx7600/` |
| M51 | done as far as the request goes (10.3): the relative statement for 1B, 3B and 8B against llama.cpp Vulkan Q4_0 and Q4_K_M, and the finding that llama.cpp does not run correctly on that driver as shipped (default: aborts; cooperative-matrix path: wrong output; correct only with that path disabled; OpenCL rejects the GPU). No stock arm. Every number is local | `m51/README.md` (statement only) |

10.2 (HIP) is not part of this task (owner decision 2026-10-09). Extras of 10.4 (real-text prompt variant, decode with and without the final configuration) are in `7900xtx/` and `rx7600/`; not done for M51.

## RX 7900 XTX in one paragraph

Tuned `4w` is ahead of llama.cpp Vulkan Q4_0 (1.36 to 1.56) and Q4_K_M (1.73 to 1.95) at every model size, by either timer and at both settings; `lc`
pays the card's clock ramp (its `clock_low` runs are counted and shown); stock, parent and tuned were measured in the same session, so the review's
cross-session objection no longer applies (tuned / stock `4w` 3.28 / 4.12 / 3.89; the 3B and 8B stock `4w` values rest on `clock_low` runs only, all eight under the 2670 MHz floor, so they have no strict value; 1B has five valid runs of seven).

**Owner decision needed:** `aggregate.py --accept clock_low` was written for the 780M; this task counts `clock_low`-only runs on the 7900 XTX (stock `4w` at 3B and 8B, `lc` Q4_0 and 1B Q4_K_M) and on the RX 7600 (8B stock `4w`) as the 780M and the pushed 2026-10-08 RX 7600 result did. No owner decision extends it to these cards; `cells-strict.csv` holds the rule as it stands. Please confirm or reject the counting rule before the figures are used. Details, deviations and audit: `7900xtx/README.md`.

## RX 7600 in one paragraph

With the round 2 final as the tuned arm, tuned `4w` is ahead of llama.cpp Vulkan Q4_0 by 1.89 / 1.74 / 1.63 (1B / 3B / 8B) and of Q4_K_M by 2.21 / 2.09 / 2.01; the
`llama-bench` slow path at 8B (prompt-only test at 2048 tokens) is shown and named, the 8B llama.cpp number is `lc`. Details: `rx7600/README.md`.

## M51 in one paragraph

Tuned ExecuTorch `4w` is ahead of llama.cpp Vulkan Q4_0 and Q4_K_M at 1B, 3B and 8B, by both timers. llama.cpp's own defaults abort on that driver and its cooperative-matrix path computes wrong output; the comparison uses the one configuration that matches llama.cpp's CPU backend. No figure is committed. Details: `m51/README.md`.
