# Roofs re-measured on the current driver (R6)

igpu-roofline (the GPU host's existing checkout and virtual environment, used as it is), `shapes --local`, then `run --local --plan quick` and `--plan fast`
(`tools/q-roof.sh`), 2026-10-09 02:59 to 03:32 UTC, AMDVLK 2025.Q2.1, DVFS-governed (clocks not pinned, as found), nothing else on the card. `REPORT.md` and
`summary.json` are copied from the run directory with the host name replaced. The confirmed medians of the fast plan are the roofs used in `proposal.md`
("percent of the roofs"): matrix fp16 with fp32 accumulate 140.834 TFLOP/s, matrix int8 141.597 TOP/s, fp16 FMA 65.044 TFLOP/s, int8 dot 70.489 TOP/s,
global read 1109.441 GB/s; matrix fp16 fed from shared 137.155 (fp32 accumulate) and int8 fed from shared 114.714. The 2026-09-27 values
(`sarc-1.5-7900xtx-4w`): 141.9 / 142.6 / 63.4 / 69.2 / 1108.
