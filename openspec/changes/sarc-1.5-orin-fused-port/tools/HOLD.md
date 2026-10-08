# HOLD: how the coordinator pauses this campaign (owner decision D6, 2026-10-06)

- Device queues: create `~/hmz-sarc-orin-fused/HOLD` on `duck-naughty`. Every tool of this campaign that starts a
  GPU job calls `hold_wait` (`common.sh`) first: `e2e5.sh` and `trace.sh` once per session (they hold the gpu-lab
  lock for the whole session: a timed session of up to 70 minutes is one job), `gl.sh` before every microbench or
  probe process, `gatelib.sh` before `verify.sh`. The running job finishes; the next one waits, polling every 30 s,
  and writes `HOLD ...` / `HOLD released ...` into its log.
- Workstation builds: create `/mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/HOLD`; `build-orin.sh` waits
  before its cross build.
- Remove the file to let the queue continue. The campaign never creates or removes `HOLD`, and its GPU-process
  guard does not learn the coordinator's processes: a foreign GPU process seen while a job of this campaign runs
  still ends that job (exit 76).
