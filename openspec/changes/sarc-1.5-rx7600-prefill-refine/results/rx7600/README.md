# results/rx7600

Evidence of the RX 7600 prefill campaign. Per-session folders are under `sessions/`; the summaries are in `../../STATUS.md` and `../../proposal.md`.

**Placeholders (coordinator order M2c, 2026-10-07 23:14 UTC).** Host names and absolute paths are replaced in every file here:
`<host>` = the GPU workstation, `<artifacts>` = the campaign's artifact directory (raw runs, logs, builds, staged binaries),
`<campaign-root>` = the campaign directory holding the `executorch` working copy, `<mesa>` = the user-space RADV install,
`<models-src>` = the directory of the six model files. **The substitution is the only edit made to those files** (the
`host` column of every `runs.csv`, `env.txt`, `STAGE.md`, `verify.meta`, `trace.out`, ...): 786 lines in 32 files, checked by
applying the same substitution to the previous committed version and comparing byte for byte (0 differences).
