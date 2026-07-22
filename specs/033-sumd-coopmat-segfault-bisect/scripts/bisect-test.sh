#!/usr/bin/env bash
# Per-commit backup->build->flash->test->restore->verdict step for
# specs/033-sumd-coopmat-segfault-bisect. See ../contracts/bisect-test-script.md for the full
# contract this implements.
#
# Usage: bisect-test.sh <sumd-worktree-dir> [bisect_role]
#
# Exit codes (matches `git bisect run`'s convention):
#   0    good  (COOPMAT_BENCH_CORRECTNESS_ONLY=1 test_coopmat_linear_bench_origcm completes all
#               16 test cases with no crash)
#   1    bad   (process crashes/stops before completing)
#   125  skip  (build failed, flash/stage failed, driver-hash drift mid-step, or a hang past the
#               bounded timeout with the device left unresponsive)
#
# Rule 0 (sumd/CLAUDE.md) is lifted workspace-wide (2026-07-17) -- but this script itself still
# never opens SUMD source; the one-time post-convergence culprit-diff read happens separately,
# outside this script, per quickstart.md Step 3.

set -u
WORKTREE="${1:?Usage: $0 <sumd-worktree-dir> [bisect_role]}"
ROLE="${2:-bisect-step}"

# --- fixed config (spec 033) ---
SSH_HOST=sj1-dmckee-d01
SERIAL=0000088f8e579c33
PIN_SCRIPT=/sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh
GPUFREQ=509000
MIFFREQ=2730000
INTFREQ=663000
DEVICE_DIR=/data/local/tmp/llama_vk
BINARY=test_coopmat_linear_bench_origcm
BACKUP_DIR=/sarc-c/gpusw/users/yanwen.xu
TEST_TIMEOUT=90

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_DIR="$(dirname "$SCRIPT_DIR")"
REPORT="${BISECT_REPORT_OVERRIDE:-$FEATURE_DIR/results/bisect-report.md}"
TOMBSTONE_DIR="$FEATURE_DIR/results/tombstones"
BACKUP_LOG="${BISECT_BACKUP_LOG_OVERRIDE:-$FEATURE_DIR/results/.driver-backup-log.tsv}"

mkdir -p "$TOMBSTONE_DIR"
touch "$BACKUP_LOG"

SHORT_SHA="$(basename "$WORKTREE")"
COMMIT_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
COMMIT_DATE="$(git -C "$WORKTREE" log -1 --format='%ci' HEAD)"

append_row() {
  # append_row <driver_hash_post_flash> <driver_hash_pre_test> <build_outcome> <verdict> <crash_evidence> <notes>
  local dhpf="$1" dhpt="$2" build_outcome="$3" verdict="$4" crash_evidence="$5" notes="$6"
  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$SHORT_SHA" "$ROLE" "$COMMIT_SHA" "$COMMIT_DATE" "$dhpf" "$dhpt" "$build_outcome" "$verdict" "$crash_evidence" "$notes" \
    >> "$REPORT"
}

log_backup() {
  # log_backup <found_hash> <backup_path> <restored_after_step>
  printf '%s\t%s\t%s\t%s\t%s\n' "$SHORT_SHA" "$COMMIT_SHA" "$1" "$2" "$3" >> "$BACKUP_LOG"
}

fail_skip() {
  local reason="$1"
  echo "commit=$COMMIT_SHA driver_hash=\"\" verdict=skip reason=\"$reason\""
  append_row "" "" "build-failed" "skip" "" "$reason"
  exit 125
}

# --- 1. build (local, on the build box) ---
echo "=== building $SHORT_SHA ($COMMIT_SHA) ==="
if ! git -C "$WORKTREE" submodule update --init --recursive; then
  fail_skip "submodule update failed"
fi

CLEAN_LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v vulkan-sdk | paste -sd:)"
BUILD_OK=1
if ! (cd "$WORKTREE" && LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" uv run scripts/run.py --os android --build --build-type release); then
  echo "build failed, retrying once (GpuRt 'Too many users' workaround already applied via stripped LD_LIBRARY_PATH)"
  if ! (cd "$WORKTREE" && LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" uv run scripts/run.py --os android --build --build-type release); then
    BUILD_OK=0
  fi
fi
[ "$BUILD_OK" = "1" ] || fail_skip "build failed twice (uv run scripts/run.py --os android --build --build-type release)"

SRC="$WORKTREE/out/android-arm64-release/vulkan.samsung.so"
[ -f "$SRC" ] || fail_skip "build reported success but $SRC is missing"

# --- 2. stage to NFS, cmp-verified ---
STAGE_DIR="/sarc-c/gpusw/users/yanwen.xu/sumd-deploy/$SHORT_SHA"
mkdir -p "$STAGE_DIR"
cp "$SRC" "$STAGE_DIR/vulkan.samsung.so"
cmp -s "$SRC" "$STAGE_DIR/vulkan.samsung.so" || fail_skip "NFS staging cmp-verify failed (share full/truncated write?)"

# --- 3. backup + flash + identify + pin + test + restore (remote, via ssh $SSH_HOST) ---
REMOTE_OUT="$(ssh "$SSH_HOST" bash -s -- "$SERIAL" "$STAGE_DIR" "$PIN_SCRIPT" "$GPUFREQ" "$MIFFREQ" "$INTFREQ" \
  "$DEVICE_DIR" "$BINARY" "$BACKUP_DIR" "$TEST_TIMEOUT" <<'REMOTE_SCRIPT' 2>&1
set -u
S="$1"; VK_DIR="$2"; PIN_SCRIPT="$3"; GPUFREQ="$4"; MIFFREQ="$5"; INTFREQ="$6"
DEVICE_DIR="$7"; BINARY="$8"; BACKUP_DIR="$9"; TEST_TIMEOUT="${10}"

fail() { echo "REMOTE_FAIL:$1"; exit 1; }

# --- 3a. backup whatever's on the device before we touch it (FR-003) ---
FOUND_HASH="$(adb -s "$S" shell md5sum /vendor/lib64/hw/vulkan.samsung.so </dev/null | awk '{print $1}')"
[ -n "$FOUND_HASH" ] || fail "could not md5sum the pre-flash on-device driver"
echo "FOUND_HASH:$FOUND_HASH"

EXISTING_BACKUP="$(ls "$BACKUP_DIR"/vulkan.samsung.so.*"$FOUND_HASH"*backup* 2>/dev/null | head -1)"
if [ -n "$EXISTING_BACKUP" ]; then
  BACKUP_PATH="$EXISTING_BACKUP"
  echo "BACKUP_REUSED:1"
else
  DATESTAMP="$(adb -s "$S" shell date +%Y%m%d </dev/null | tr -d '\r')"
  BACKUP_PATH="$BACKUP_DIR/vulkan.samsung.so.$FOUND_HASH-backup-${DATESTAMP:-unknown-date}"
  adb -s "$S" pull /vendor/lib64/hw/vulkan.samsung.so "$BACKUP_PATH" </dev/null >/dev/null 2>&1 || fail "adb pull of pre-existing driver failed"
  echo "BACKUP_REUSED:0"
fi
echo "BACKUP_PATH:$BACKUP_PATH"

# --- 3b. flash candidate ---
adb -s "$S" root </dev/null >/dev/null 2>&1
adb -s "$S" remount </dev/null >/dev/null 2>&1
adb -s "$S" shell setenforce 0 </dev/null
adb -s "$S" shell stop </dev/null
adb -s "$S" shell mkdir -p /data/local/tmp/hw64 </dev/null
adb -s "$S" shell chmod 777 /data/local/tmp/hw64 </dev/null
adb -s "$S" push "$VK_DIR/vulkan.samsung.so" /vendor/lib64/hw/vulkan.samsung.so </dev/null || fail "adb push of candidate driver failed"
adb -s "$S" shell chmod 644 /vendor/lib64/hw/vulkan.samsung.so </dev/null
adb -s "$S" shell chmod 777 /data/local/tmp </dev/null
adb -s "$S" remount </dev/null >/dev/null 2>&1
adb -s "$S" shell start </dev/null
sleep 5
adb -s "$S" wait-for-device </dev/null

DRIVER_HASH_POST_FLASH="$(adb -s "$S" shell md5sum /vendor/lib64/hw/vulkan.samsung.so </dev/null | awk '{print $1}')"
echo "DRIVER_HASH_POST_FLASH:$DRIVER_HASH_POST_FLASH"
[ -n "$DRIVER_HASH_POST_FLASH" ] || fail "driver md5sum came back empty after flash"

# --- 3c. pin clocks (workspace default for this device) ---
S="$S" GPUFREQ="$GPUFREQ" MIFFREQ="$MIFFREQ" INTFREQ="$INTFREQ" "$PIN_SCRIPT" </dev/null
READBACK="$(adb -s "$S" shell "cat /sys/class/devfreq/23400000.sgpu/max_freq /sys/class/devfreq/17000010.devfreq_mif/cur_freq /sys/class/devfreq/17000020.devfreq_int/cur_freq" </dev/null | tr -d '\r')"
GOT_GPU_MAX="$(echo "$READBACK" | sed -n 1p)"; GOT_MIF="$(echo "$READBACK" | sed -n 2p)"; GOT_INT="$(echo "$READBACK" | sed -n 3p)"
if [ "$GOT_GPU_MAX" != "$GPUFREQ" ] || [ "$GOT_MIF" != "$MIFFREQ" ] || [ "$GOT_INT" != "$INTFREQ" ]; then
  S="$S" GPUFREQ="$GPUFREQ" MIFFREQ="$MIFFREQ" INTFREQ="$INTFREQ" "$PIN_SCRIPT" </dev/null
  READBACK2="$(adb -s "$S" shell "cat /sys/class/devfreq/23400000.sgpu/max_freq /sys/class/devfreq/17000010.devfreq_mif/cur_freq /sys/class/devfreq/17000020.devfreq_int/cur_freq" </dev/null | tr -d '\r')"
  G2="$(echo "$READBACK2" | sed -n 1p)"; M2="$(echo "$READBACK2" | sed -n 2p)"; I2="$(echo "$READBACK2" | sed -n 3p)"
  [ "$G2" = "$GPUFREQ" ] && [ "$M2" = "$MIFFREQ" ] && [ "$I2" = "$INTFREQ" ] || fail "clock pin did not verify after retry (got GPU_max=$G2 MIF=$M2 INT=$I2)"
fi

DRIVER_HASH_PRE_TEST="$(adb -s "$S" shell md5sum /vendor/lib64/hw/vulkan.samsung.so </dev/null | awk '{print $1}')"
echo "DRIVER_HASH_PRE_TEST:$DRIVER_HASH_PRE_TEST"
if [ "$DRIVER_HASH_PRE_TEST" != "$DRIVER_HASH_POST_FLASH" ]; then
  fail "driver hash drifted mid-step (post-flash=$DRIVER_HASH_POST_FLASH pre-test=$DRIVER_HASH_PRE_TEST)"
fi

# --- 3d. run the correctness bench under a bounded timeout ---
BEFORE_LATEST_TS="$(adb -s "$S" shell "ls -1t /data/tombstones/ 2>/dev/null | head -1" </dev/null | tr -d '\r')"

RUN_OUT="$(adb -s "$S" shell "cd $DEVICE_DIR && COOPMAT_BENCH_CORRECTNESS_ONLY=1 timeout $TEST_TIMEOUT ./$BINARY; echo EXITCODE:\$?" </dev/null 2>&1)"
EXIT_CODE="$(echo "$RUN_OUT" | grep -o 'EXITCODE:[0-9]*' | tail -1 | cut -d: -f2)"
CONSOLE="$(echo "$RUN_OUT" | grep -v '^EXITCODE:')"
LAST_LINE="$(echo "$CONSOLE" | tail -1 | tr -d '\r' | tr '\t' ' ')"

if echo "$CONSOLE" | grep -q "Completed 16 test cases" && [ "${EXIT_CODE:-1}" = "0" ]; then
  VERDICT="good"
  SIGNATURE=""
elif [ "${EXIT_CODE:-}" = "124" ]; then
  if adb -s "$S" get-state </dev/null 2>&1 | grep -q device; then
    VERDICT="bad"
    SIGNATURE="hang (timeout ${TEST_TIMEOUT}s exceeded, device still responsive)"
  else
    VERDICT="skip"
    SIGNATURE="hang, device unresponsive after timeout -- needs manual recovery"
  fi
else
  VERDICT="bad"
  SIGNATURE="crash exit_code=${EXIT_CODE:-unknown}"
fi
echo "VERDICT:$VERDICT"
echo "EXIT_CODE:${EXIT_CODE:-}"
echo "SIGNATURE:$SIGNATURE"
echo "LAST_LINE:$LAST_LINE"

if [ "$VERDICT" = "bad" ]; then
  AFTER_LATEST_TS="$(adb -s "$S" shell "ls -1t /data/tombstones/ 2>/dev/null | head -1" </dev/null | tr -d '\r')"
  if [ -n "$AFTER_LATEST_TS" ] && [ "$AFTER_LATEST_TS" != "$BEFORE_LATEST_TS" ]; then
    adb -s "$S" pull "/data/tombstones/$AFTER_LATEST_TS" "$VK_DIR/tombstone-$AFTER_LATEST_TS.txt" </dev/null >/dev/null 2>&1
    echo "TOMBSTONE_NFS:$VK_DIR/tombstone-$AFTER_LATEST_TS.txt"
    SIGNAL_LINE="$(grep -m1 'signal ' "$VK_DIR/tombstone-$AFTER_LATEST_TS.txt" 2>/dev/null | tr -d '\r')"
    echo "TOMBSTONE_SIGNAL:$SIGNAL_LINE"
  else
    echo "TOMBSTONE_NFS:"
  fi
fi

# --- 3e. restore whatever was on the device before this step (FR-003) ---
if [ "$VERDICT" != "skip" ] || echo "${SIGNATURE:-}" | grep -q "^hang (timeout"; then
  adb -s "$S" root </dev/null >/dev/null 2>&1
  adb -s "$S" remount </dev/null >/dev/null 2>&1
  adb -s "$S" shell stop </dev/null
  adb -s "$S" push "$BACKUP_PATH" /vendor/lib64/hw/vulkan.samsung.so </dev/null || fail "restore push failed"
  adb -s "$S" shell chmod 644 /vendor/lib64/hw/vulkan.samsung.so </dev/null
  adb -s "$S" remount </dev/null >/dev/null 2>&1
  adb -s "$S" shell start </dev/null
  sleep 5
  adb -s "$S" wait-for-device </dev/null
  RESTORED_HASH="$(adb -s "$S" shell md5sum /vendor/lib64/hw/vulkan.samsung.so </dev/null | awk '{print $1}')"
  echo "RESTORED_HASH:$RESTORED_HASH"
  if [ "$RESTORED_HASH" = "$FOUND_HASH" ]; then
    echo "RESTORED_OK:1"
  else
    echo "RESTORED_OK:0"
  fi
else
  echo "RESTORED_OK:skipped-device-unresponsive"
fi
REMOTE_SCRIPT
)"

echo "--- remote output ---"
echo "$REMOTE_OUT"
echo "--- end remote output ---"

if [ -z "$REMOTE_OUT" ]; then
  fail_skip "ssh to $SSH_HOST produced no output at all (connection/auth failure?)"
fi

if echo "$REMOTE_OUT" | grep -q '^REMOTE_FAIL:'; then
  REASON="$(echo "$REMOTE_OUT" | grep '^REMOTE_FAIL:' | head -1 | cut -d: -f2-)"
  DHPF="$(echo "$REMOTE_OUT" | grep '^DRIVER_HASH_POST_FLASH:' | head -1 | cut -d: -f2)"
  DHPT="$(echo "$REMOTE_OUT" | grep '^DRIVER_HASH_PRE_TEST:' | head -1 | cut -d: -f2)"
  append_row "${DHPF:-}" "${DHPT:-}" "flash-failed" "skip" "" "$REASON"
  echo "commit=$COMMIT_SHA verdict=skip reason=\"$REASON\""
  exit 125
fi

FOUND_HASH="$(echo "$REMOTE_OUT" | grep '^FOUND_HASH:' | head -1 | cut -d: -f2)"
BACKUP_PATH="$(echo "$REMOTE_OUT" | grep '^BACKUP_PATH:' | head -1 | cut -d: -f2-)"
BACKUP_REUSED="$(echo "$REMOTE_OUT" | grep '^BACKUP_REUSED:' | head -1 | cut -d: -f2)"
DHPF="$(echo "$REMOTE_OUT" | grep '^DRIVER_HASH_POST_FLASH:' | head -1 | cut -d: -f2)"
DHPT="$(echo "$REMOTE_OUT" | grep '^DRIVER_HASH_PRE_TEST:' | head -1 | cut -d: -f2)"
VERDICT="$(echo "$REMOTE_OUT" | grep '^VERDICT:' | head -1 | cut -d: -f2)"
EXIT_CODE="$(echo "$REMOTE_OUT" | grep '^EXIT_CODE:' | head -1 | cut -d: -f2)"
SIGNATURE="$(echo "$REMOTE_OUT" | grep '^SIGNATURE:' | head -1 | cut -d: -f2-)"
LAST_LINE="$(echo "$REMOTE_OUT" | grep '^LAST_LINE:' | head -1 | cut -d: -f2-)"
TOMBSTONE_NFS="$(echo "$REMOTE_OUT" | grep '^TOMBSTONE_NFS:' | head -1 | cut -d: -f2-)"
TOMBSTONE_SIGNAL="$(echo "$REMOTE_OUT" | grep '^TOMBSTONE_SIGNAL:' | head -1 | cut -d: -f2-)"
RESTORED_OK="$(echo "$REMOTE_OUT" | grep '^RESTORED_OK:' | head -1 | cut -d: -f2)"

log_backup "$FOUND_HASH" "$BACKUP_PATH" "$RESTORED_OK"

if [ -z "$DHPF" ] || [ -z "$DHPT" ] || [ -z "$VERDICT" ]; then
  fail_skip "one or more expected fields missing from remote output (driver_hash_post_flash='$DHPF' driver_hash_pre_test='$DHPT' verdict='$VERDICT') -- treating as skip, not a verdict"
fi

CRASH_EVIDENCE=""
if [ "$VERDICT" = "bad" ]; then
  if [ -n "$TOMBSTONE_NFS" ]; then
    LOCAL_TOMBSTONE="$TOMBSTONE_DIR/$SHORT_SHA.txt"
    cp "$TOMBSTONE_NFS" "$LOCAL_TOMBSTONE" 2>/dev/null
    CRASH_EVIDENCE="tombstone=results/tombstones/$SHORT_SHA.txt signal=\"${TOMBSTONE_SIGNAL:-}\" exit_code=$EXIT_CODE last_line=\"$LAST_LINE\""
  else
    CRASH_EVIDENCE="no tombstone (fallback) exit_code=$EXIT_CODE last_line=\"$LAST_LINE\""
  fi
fi

NOTES="$SIGNATURE"
if [ "$BACKUP_REUSED" = "1" ]; then
  NOTES="$NOTES backup_reused=$BACKUP_PATH"
else
  NOTES="$NOTES backed_up_to=$BACKUP_PATH"
fi
NOTES="$NOTES restored_ok=$RESTORED_OK"

append_row "$DHPF" "$DHPT" "success" "$VERDICT" "$CRASH_EVIDENCE" "$NOTES"
echo "commit=$COMMIT_SHA driver_hash=\"$DHPF\" verdict=$VERDICT ${SIGNATURE:+reason=\"$SIGNATURE\"} ${TOMBSTONE_NFS:+tombstone=\"$TOMBSTONE_DIR/$SHORT_SHA.txt\"}"

case "$VERDICT" in
  good) exit 0 ;;
  bad) exit 1 ;;
  *) exit 125 ;;
esac
