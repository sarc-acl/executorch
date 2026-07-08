#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# specs/023-8da4w-int8-dbuf-sweep Foundational T009: drives
# test_dq8ca_dbuf_sweep_bench on M5 EVT1, one process per
# ET_VK_DQ8CA_COOPMAT_VARIANT value (research.md Decision 2 -- an Xclipse
# PAL pipeline-creation crash on one variant cannot be caught in-process, so
# isolation is at the process/variant boundary). Run from anywhere; requires
# ssh access to the adb host and the binary already pushed or locatable.
#
# Usage:
#   run_dbuf_sweep.sh [--correctness-only] <dbuf1|dbuf2|dbuf3|dbuf4> [dbuf...]
#
# Env (override the defaults below, e.g. after a re-flash or host change --
# see .shared-context/instruction-for-ai/README.md §Conventions):
#   HOST, S (adb serial), D (on-device dir), LOCAL_BIN (host path to the
#   built test_dq8ca_dbuf_sweep_bench binary)
#
# Exit code: 0 if every requested variant's invocation exited 0, 1 otherwise
# (per-variant results are still all recorded either way -- FR-004: a failed
# variant must be reported, not silently dropped).

set -u

HOST="${HOST:-yanwen.xu@sj1-dmckee-d01}"
S="${S:-0000088f8e579c33}"
D="${D:-/data/local/tmp/llama_vk}"
LOCAL_BIN="${LOCAL_BIN:-}"
BIN_NAME="test_dq8ca_dbuf_sweep_bench"

CORRECTNESS_ONLY=0
VARIANTS=()
for arg in "$@"; do
  case "$arg" in
    --correctness-only) CORRECTNESS_ONLY=1 ;;
    dbuf1|dbuf2|dbuf3|dbuf4) VARIANTS+=("$arg") ;;
    *)
      echo "usage: $0 [--correctness-only] <dbuf1|dbuf2|dbuf3|dbuf4> [dbuf...]" >&2
      exit 2
      ;;
  esac
done
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  echo "usage: $0 [--correctness-only] <dbuf1|dbuf2|dbuf3|dbuf4> [dbuf...]" >&2
  exit 2
fi

RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/raw"
mkdir -p "$RESULTS_DIR"

if [ -n "$LOCAL_BIN" ]; then
  echo "[push] $LOCAL_BIN -> $HOST:$D/$BIN_NAME"
  scp "$LOCAL_BIN" "$HOST:$D/$BIN_NAME" || { echo "[push] FAILED"; exit 1; }
  ssh "$HOST" "adb -s $S push $D/$BIN_NAME $D/$BIN_NAME 2>/dev/null; adb -s $S shell chmod 755 $D/$BIN_NAME" \
    || { echo "[push] adb push/chmod FAILED"; exit 1; }
fi

OVERALL_STATUS=0
for variant in "${VARIANTS[@]}"; do
  LOG="$RESULTS_DIR/${variant}$([ "$CORRECTNESS_ONLY" = 1 ] && echo _correctness_only).log"
  ENV_PREFIX="ET_VK_DQ8CA_COOPMAT_VARIANT=$variant"
  if [ "$CORRECTNESS_ONLY" = 1 ]; then
    ENV_PREFIX="$ENV_PREFIX DQ8CA_DBUF_SWEEP_CORRECTNESS_ONLY=1"
  fi
  echo "[run] variant=$variant correctness_only=$CORRECTNESS_ONLY -> $LOG"
  # One process per variant (research.md Decision 2): each ssh+adb shell
  # invocation below is a fresh process on-device, so a pipeline-creation
  # crash in one variant cannot corrupt another's results.
  ssh "$HOST" "adb -s $S shell \"cd $D && $ENV_PREFIX ./$BIN_NAME\"" > "$LOG" 2>&1
  exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "[run] variant=$variant FAILED (exit $exit_code) -- see $LOG" | tee -a "$LOG"
    echo "pipeline_crash_or_failure exit_code=$exit_code" >> "$LOG"
    OVERALL_STATUS=1
  else
    echo "[run] variant=$variant OK"
  fi
done

exit "$OVERALL_STATUS"
