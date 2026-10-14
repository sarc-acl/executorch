# Candidate 2: SDPA error against the fp64 reference (D3 criterion 1)

One coherent run: `tools/sdpa_evidence.sh c2-fused` (all tiers, both arms, one session, 2026-10-07 23:14 to 23:49 UTC,
under the device lock), `test_llama_microbench --sdpa-correctness-only --sdpa-tier=<all|extended|peaked|full>` with
`ET_VK_SDPA_ERROR_REPORT=1`. Parent arm = candidate 1 (`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7`, three
kernels); candidate arm = parent plus `ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko`.
Both arms: the same binary (`stage/c2-fused/test_llama_microbench`, sha256 `21cfb63d...`).

- `error.csv`: the complete table, 17 rows (extended 8, peaked 5, full 4), generated from `logs/*.log` by
  `tools/sdpa_error_table.py`. Nothing removed.
- `runs.txt`: rc and pass counts of the eight runs (all rc 0, 0 failed).
- `bitwise.txt`: raw fp16 outputs compared byte for byte: 21 of 21 differ (expected: a different kernel and
  accumulation order; this candidate is judged by D3, not by bit identity).
- `logs/`: the raw logs behind every value.

Production criterion (D3.1: S = 2048, every head configuration; rms and maximum error not larger than the parent's):
tier `full` 4 of 4 `yes` (1B, 3B, 8B at S = 2048, 8B at S = 1024); `extended` 8 of 8 `yes`; `peaked` S = 2048: 2 of 2 `yes`
(max ratio 1.000 and 0.930).

One row reads `NO`: `peaked_tiny_gqa_s256` (S = 256, tiny GQA test shape, peaked input): maximum error 1.9819e-03
against the parent's 1.7378e-03 (ratio 1.140). Its rms error is lower (2.2077e-04 against 2.7172e-04, ratio 0.812), and
all other peaked rows have a maximum ratio of 0.93 to 1.00. It is not a production shape (S = 256, a correctness-test
shape), so D3.1 does not cover it; it is reported here as it is, not as a pass. A single-element maximum of one
test shape that the fp16 parent happens to round favourably is the reading consistent with the rms; UNVERIFIED
(no per-element analysis was made).
