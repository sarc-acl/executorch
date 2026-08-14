#!/usr/bin/env bash
# Per-commit build+flash+measure+verdict step, forked from
# specs/032-sumd-driver-bisect/scripts/bisect-test.sh for
# openspec/changes/verify-gfxsw76300-dot4-patch (GFXSW-76300 dot4-patch verification).
# Kept as its own copy so it doesn't append to specs/032's historical bisect-report.md.
#
# Usage: verify-test.sh <sumd-worktree-dir> [role]
#
# Exit codes (matches `git bisect run`'s convention, inherited from the original script):
#   0   good  (8da4w prefill tok/s > 4w prefill tok/s)
#   1   bad   (otherwise)
#   125 skip  (build failed, flash failed, driver crashed/hung, or clock pin didn't verify)

set -u
WORKTREE="${1:?Usage: $0 <sumd-worktree-dir> [role]}"
ROLE="${2:-verify-step}"

# --- fixed config ---
SSH_HOST=xgpusw-debug07
SERIAL=00000bb7cc34abd3   # corrected M41 serial (2026-08-12) -- specs/032's 00000a34cdd4abd3 is stale
PIN_SCRIPT=/sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh
GPUFREQ=980000
MIFFREQ=5333000
INTFREQ=800000
DEVICE_DIR=/data/local/tmp/llama_vk
PTE_4W=llama3_2_1b_4w_texture_ctx3072.pte
PTE_8DA4W=llama3_2_1b_8da4w_texture_ctx3072.pte
TOKENIZER=tokenizer.model
PROMPT_FILE=p2048_exact.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_DIR="$(dirname "$SCRIPT_DIR")"
REPORT="${BISECT_REPORT_OVERRIDE:-$FEATURE_DIR/results/bisect-report.md}"

SHORT_SHA="$(basename "$WORKTREE")"
COMMIT_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
COMMIT_DATE="$(git -C "$WORKTREE" log -1 --format='%ci' HEAD)"

append_row() {
  # append_row <driver_version> <build_outcome> <4w> <8da4w> <verdict> <notes>
  local driver_version="$1" build_outcome="$2" v4w="$3" v8da4w="$4" verdict="$5" notes="$6"
  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$SHORT_SHA" "$ROLE" "$COMMIT_SHA" "$COMMIT_DATE" "$driver_version" "$v4w" "$v8da4w" "$verdict" "$notes" \
    >> "$REPORT"
}

fail_skip() {
  local reason="$1"
  echo "commit=$COMMIT_SHA driver_version=\"\" 4w= 8da4w= verdict=skip reason=\"$reason\""
  append_row "" "build-failed" "" "" "skip" "$reason"
  exit 125
}

# --- 1. build (local, on the build box) ---
echo "=== building $SHORT_SHA ($COMMIT_SHA) ==="
if ! git -C "$WORKTREE" submodule update --init --recursive; then
  fail_skip "submodule update failed"
fi

CLEAN_LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v vulkan-sdk | paste -sd:)"
if ! (cd "$WORKTREE" && LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" uv run scripts/run.py --os android --build --build-type release); then
  fail_skip "build failed (uv run scripts/run.py --os android --build --build-type release)"
fi

SRC="$WORKTREE/out/android-arm64-release/vulkan.samsung.so"
if [ ! -f "$SRC" ]; then
  fail_skip "build reported success but $SRC is missing"
fi

# --- 2. stage to NFS, cmp-verified ---
STAGE_DIR="/sarc-c/gpusw/users/yanwen.xu/sumd-deploy/$SHORT_SHA"
mkdir -p "$STAGE_DIR"
cp "$SRC" "$STAGE_DIR/vulkan.samsung.so"
if ! cmp -s "$SRC" "$STAGE_DIR/vulkan.samsung.so"; then
  fail_skip "NFS staging cmp-verify failed (share full/truncated write?)"
fi

# --- 3. flash + identify + pin + measure (remote, via ssh $SSH_HOST) ---
# NOTE: every adb/pin_freqs.sh invocation below redirects stdin from /dev/null. Without this,
# `adb shell`/`adb push`/pin_freqs.sh's own internal adb calls can consume bytes from this
# heredoc's stream (which is *also* this remote bash -s process's stdin), silently truncating
# the rest of this script — reproduced empirically 2026-07-16 during T009's dry run: the script
# hung right after the first bare `adb shell setenforce 0` with no error, no further output, and
# no trace of the remaining commands ever running. `< /dev/null` on the command itself (not e.g.
# `exec < /dev/null` at the top, which was also tried and instead prevented bash from reading the
# *rest of its own script* from the same heredoc) fixes it without changing behavior otherwise.
REMOTE_OUT="$(ssh "$SSH_HOST" bash -s -- "$SERIAL" "$STAGE_DIR" "$PIN_SCRIPT" "$GPUFREQ" "$MIFFREQ" "$INTFREQ" \
  "$DEVICE_DIR" "$PTE_4W" "$PTE_8DA4W" "$TOKENIZER" "$PROMPT_FILE" <<'REMOTE_SCRIPT' 2>&1
set -u
S="$1"; VK_DIR="$2"; PIN_SCRIPT="$3"; GPUFREQ="$4"; MIFFREQ="$5"; INTFREQ="$6"
DEVICE_DIR="$7"; PTE_4W="$8"; PTE_8DA4W="$9"; TOKENIZER="${10}"; PROMPT_FILE="${11}"

fail() { echo "REMOTE_FAIL:$1"; exit 1; }

adb -s "$S" root </dev/null >/dev/null 2>&1
adb -s "$S" remount </dev/null >/dev/null 2>&1
adb -s "$S" shell setenforce 0 </dev/null
adb -s "$S" shell stop </dev/null
adb -s "$S" shell mkdir -p /data/local/tmp/hw64 </dev/null
adb -s "$S" shell chmod 777 /data/local/tmp/hw64 </dev/null
adb -s "$S" push "$VK_DIR/vulkan.samsung.so" /vendor/lib64/hw/vulkan.samsung.so </dev/null || fail "adb push failed"
adb -s "$S" shell chmod 644 /vendor/lib64/hw/vulkan.samsung.so </dev/null
adb -s "$S" shell chmod 777 /data/local/tmp </dev/null
adb -s "$S" remount </dev/null >/dev/null 2>&1
adb -s "$S" shell start </dev/null
sleep 5
adb -s "$S" wait-for-device </dev/null

DRIVER_MD5="$(adb -s "$S" shell md5sum /vendor/lib64/hw/vulkan.samsung.so </dev/null | awk '{print $1}')"
echo "DRIVER_MD5:$DRIVER_MD5"
[ -n "$DRIVER_MD5" ] || fail "driver md5sum came back empty after flash"

S="$S" GPUFREQ="$GPUFREQ" MIFFREQ="$MIFFREQ" INTFREQ="$INTFREQ" "$PIN_SCRIPT" </dev/null
READBACK="$(adb -s "$S" shell "cat /sys/class/devfreq/23400000.sgpu/min_freq /sys/class/devfreq/23400000.sgpu/max_freq /sys/class/devfreq/17000010.devfreq_mif/cur_freq /sys/class/devfreq/17000020.devfreq_int/cur_freq" </dev/null | tr -d '\r')"
echo "PIN_READBACK:$READBACK"
GOT_GPU_MIN="$(echo "$READBACK" | sed -n 1p)"; GOT_GPU_MAX="$(echo "$READBACK" | sed -n 2p)"
GOT_MIF="$(echo "$READBACK" | sed -n 3p)"; GOT_INT="$(echo "$READBACK" | sed -n 4p)"
if [ "$GOT_GPU_MAX" != "$GPUFREQ" ] || [ "$GOT_MIF" != "$MIFFREQ" ] || [ "$GOT_INT" != "$INTFREQ" ]; then
  # one re-pin attempt before giving up
  S="$S" GPUFREQ="$GPUFREQ" MIFFREQ="$MIFFREQ" INTFREQ="$INTFREQ" "$PIN_SCRIPT" </dev/null
  READBACK2="$(adb -s "$S" shell "cat /sys/class/devfreq/23400000.sgpu/max_freq /sys/class/devfreq/17000010.devfreq_mif/cur_freq /sys/class/devfreq/17000020.devfreq_int/cur_freq" </dev/null | tr -d '\r')"
  G2="$(echo "$READBACK2" | sed -n 1p)"; M2="$(echo "$READBACK2" | sed -n 2p)"; I2="$(echo "$READBACK2" | sed -n 3p)"
  [ "$G2" = "$GPUFREQ" ] && [ "$M2" = "$MIFFREQ" ] && [ "$I2" = "$INTFREQ" ] || fail "clock pin did not verify after retry (got GPU_max=$G2 MIF=$M2 INT=$I2)"
fi

run_once() {
  local pte="$1"
  adb -s "$S" shell "cd $DEVICE_DIR && ./llama_main_rel1.3 \
    --model_path=$DEVICE_DIR/$pte --tokenizer_path=$DEVICE_DIR/$TOKENIZER \
    --prompt_file=$DEVICE_DIR/$PROMPT_FILE --num_bos=1 --max_new_tokens=1 --ignore_eos \
    --temperature=0 --warmup=true" </dev/null 2>&1
}

OUT_4W="$(run_once "$PTE_4W")"
if echo "$OUT_4W" | grep -qE "libc\+\+abi|VK_ERROR|SIGSEGV|Fatal signal|Aborted|terminate called"; then
  echo "$OUT_4W"
  fail "4w run crashed"
fi
PREFILL_4W="$(echo "$OUT_4W" | grep -o '"prefill_token_per_sec":[0-9.]*' | head -1 | cut -d: -f2)"
[ -n "$PREFILL_4W" ] || { echo "$OUT_4W"; fail "could not parse 4w prefill_token_per_sec"; }
echo "PREFILL_4W:$PREFILL_4W"

OUT_8DA4W="$(run_once "$PTE_8DA4W")"
if echo "$OUT_8DA4W" | grep -qE "libc\+\+abi|VK_ERROR|SIGSEGV|Fatal signal|Aborted|terminate called"; then
  echo "$OUT_8DA4W"
  fail "8da4w run crashed"
fi
PREFILL_8DA4W="$(echo "$OUT_8DA4W" | grep -o '"prefill_token_per_sec":[0-9.]*' | head -1 | cut -d: -f2)"
[ -n "$PREFILL_8DA4W" ] || { echo "$OUT_8DA4W"; fail "could not parse 8da4w prefill_token_per_sec"; }
echo "PREFILL_8DA4W:$PREFILL_8DA4W"
REMOTE_SCRIPT
)"
SSH_EXIT=$?

echo "--- remote output ---"
echo "$REMOTE_OUT"
echo "--- end remote output ---"

if [ -z "$REMOTE_OUT" ]; then
  fail_skip "ssh to $SSH_HOST produced no output at all (connection/auth failure? exit=$SSH_EXIT)"
fi

if echo "$REMOTE_OUT" | grep -q '^REMOTE_FAIL:'; then
  REASON="$(echo "$REMOTE_OUT" | grep '^REMOTE_FAIL:' | head -1 | cut -d: -f2-)"
  DRIVER_MD5="$(echo "$REMOTE_OUT" | grep '^DRIVER_MD5:' | head -1 | cut -d: -f2)"
  append_row "${DRIVER_MD5:-}" "crashed-on-device" "" "" "skip" "$REASON"
  echo "commit=$COMMIT_SHA driver_version=\"${DRIVER_MD5:-}\" 4w= 8da4w= verdict=skip reason=\"$REASON\""
  exit 125
fi

DRIVER_MD5="$(echo "$REMOTE_OUT" | grep '^DRIVER_MD5:' | head -1 | cut -d: -f2)"
PREFILL_4W="$(echo "$REMOTE_OUT" | grep '^PREFILL_4W:' | head -1 | cut -d: -f2)"
PREFILL_8DA4W="$(echo "$REMOTE_OUT" | grep '^PREFILL_8DA4W:' | head -1 | cut -d: -f2)"

# Defense in depth: never let a missing value silently become a "bad" verdict — this exact
# failure mode (empty fields, no REMOTE_FAIL marker, silently computed as verdict=bad) is what
# T009's first dry run actually produced, caused by the stdin-corruption bug documented above.
if [ -z "$DRIVER_MD5" ] || [ -z "$PREFILL_4W" ] || [ -z "$PREFILL_8DA4W" ]; then
  fail_skip "one or more expected fields missing from remote output (driver_md5='$DRIVER_MD5' 4w='$PREFILL_4W' 8da4w='$PREFILL_8DA4W') — treating as skip, not a verdict"
fi

# --- 4. verdict: strict comparison, no tie-break (spec Clarifications) ---
IS_GOOD="$(awk -v a="$PREFILL_8DA4W" -v b="$PREFILL_4W" 'BEGIN{print (a>b)?1:0}')"
if [ "$IS_GOOD" = "1" ]; then
  VERDICT="good"
else
  VERDICT="bad"
fi

append_row "$DRIVER_MD5" "success" "$PREFILL_4W" "$PREFILL_8DA4W" "$VERDICT" ""
echo "commit=$COMMIT_SHA driver_version=\"$DRIVER_MD5\" 4w=$PREFILL_4W 8da4w=$PREFILL_8DA4W verdict=$VERDICT"

[ "$VERDICT" = "good" ] && exit 0 || exit 1
