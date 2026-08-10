# INVALID — do not read these as results

The first texture-IO sweep attempt (2026-08-09). Aborted by a **board reboot**
caused by the Csh shared-memory overflow described in `../TEXTURE-IO-FINDINGS.md`
§2: `tsweep_dbuf4_t128x256k32g18s32` needs 119808 B of LDS against a 65536 B
limit and hung the GPU instead of failing pipeline creation.

The reboot killed adb, so every token after it returned nothing in ~0.5 s and was
recorded as `no_results` with `exit_code: -1`. Concretely:

- `q4gsw`: 4 ok, 1 gate fail, 43 no_results
- `dq8ca`: 112 no_results (the reboot happened before this block started)

Kept only as evidence of the incident. The runtime now rejects oversized texture
tiles, so this failure mode cannot recur.
