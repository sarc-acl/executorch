#!/bin/bash
# build-stock.sh <campaign tools dir> <out root>
# Stock ExecuTorch arm: upstream release/1.5 at 985c1ceccc plus the compile-only backport of
# sarc-1.5-e2e-benchmark (kit/patches/stock-backport-03f41d2031.patch), exactly the stock build of that
# benchmark. The tree is an export of the commit and its pinned submodules from the git object stores
# (the campaign's host.sh export_commit), built in the campaign's build container with its build.sh
# (--llama --no-tests). Holds the host's build lock exclusively: never during a measurement.
#   <campaign tools dir>  tools/ of the device's sarc-1.5-<gpu>-prefill-refine change (host.sh)
#   <out root>            receives src/stock/executorch, build/stock, build/stock.src.txt
set -uo pipefail
TOOLS_DIR=$(readlink -f "$1"); OUT=$(readlink -f "$2"); COMMIT=985c1ceccc
. "$TOOLS_DIR/host.sh"
build_exclusive || exit 75
unset TMPDIR
PATCH=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/patches/stock-backport-03f41d2031.patch
SRC=$OUT/src/stock/executorch; P=$OUT/build/stock.src.txt
[[ -e $SRC || -e $P ]] && { echo "stock build already exists under $OUT" >&2; exit 2; }
mkdir -p "$OUT/src/stock" "$OUT/build"
SHA=$(git -C "$ET" rev-parse --verify "$COMMIT^{commit}") || exit 2
export B580_EXPORT_MANIFEST=$OUT/src/stock.export-manifest XE2_EXPORT_MANIFEST=$OUT/src/stock.export-manifest
export_commit "$ET" "$SHA" "$SRC" || { echo "export failed" >&2; exit 2; }
( cd "$SRC" && patch -p1 --no-backup-if-mismatch < "$PATCH" ) || { echo "backport patch failed" >&2; exit 2; }
{ echo "tag=stock"; echo "commit=$SHA"; echo "patch=$(sha256sum < "$PATCH" | cut -c1-64) stock-backport-03f41d2031.patch"
  echo "export_manifest=$(wc -l < "$B580_EXPORT_MANIFEST") trees"; echo "start=$(date -u +%FT%TZ)"; } > "$P"
SARC_MOUNT_ROOT=$(dirname "$OUT") "$ET/sarc/tools/build.sh" --llama --no-tests "$SRC" "$OUT/build/stock" > "$OUT/build/stock.log" 2>&1; rc=$?
grep -q SARC_BUILD_OK "$OUT/build/stock.log" || rc=${rc/#0/1}
{ echo "rc=$rc"; sha256sum "$OUT/build/stock/llama/examples/models/llama/llama_main" 2>&1; echo "end=$(date -u +%FT%TZ)"
  [[ $rc == 0 ]] && echo STOCK_BUILD_OK || echo STOCK_BUILD_FAILED; } >> "$P"
tail -4 "$P"; exit $rc
