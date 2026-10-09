# results/rx7600

Evidence of the RX 7600 prefill campaign. Per-session folders are under `sessions/`; the summaries are in `../../STATUS.md` and `../../proposal.md`.

**Placeholders (coordinator order M2c, 2026-10-07 23:14 UTC).** Host names and absolute paths are replaced in every file here:
`<host>` = the GPU workstation, `<artifacts>` = the campaign's artifact directory (raw runs, logs, builds, staged binaries),
`<campaign-root>` = the campaign directory holding the `executorch` working copy, `<mesa>` = the user-space RADV install,
`<models-src>` = the directory of the six model files. **The substitution is the only edit made to those files** (the
`host` column of every `runs.csv`, `env.txt`, `STAGE.md`, `verify.meta`, `trace.out`, ...): 786 lines in 32 files, checked by
applying the same substitution to the previous committed version and comparing byte for byte (0 differences).

**Round 2 (2026-10-08 23:38 UTC onwards, the linear kernels).** `round2/` holds the diagnosis and screens (`round2/README.md`), `sessions/r2-c6-pitch`
(candidate 1), `sessions/r2-c7-q4` (candidate 2), `sessions/r2-final` and `sessions/r2-final-r1` (final verification and final sessions). Two more
placeholders are used there: `<scratch>` = the local scratch disk that holds the large exports, builds and stage directories of round 2 (the root
filesystem filled up twice during round 2), `<vulkan-sdk>` = the Vulkan SDK of the native builds, `<home>` = the user's home directory. The
substitution is done by `tools/collect_session.sh` while the files are copied; nothing else of a collected file is edited.
