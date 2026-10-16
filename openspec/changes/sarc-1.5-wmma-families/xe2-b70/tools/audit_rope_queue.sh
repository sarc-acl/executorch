#!/bin/bash
# CPU-only reuse of the delivered recipe; preserve the GPU resource blocker.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
pair_lock excl
coordinator_hold 'CPU audit of delivered partial RoPE recipe'
python3 -B "$TOOLS/audit_rope_bundle.py" \
  --bundle "$A/input-preparation/b580-rope-partial-v1" \
  --archive "$A/intel-rope-partial-bundle.tar.gz" \
  --original-root "$XE2_MODEL_ROOT" \
  --output "$A/input-preparation/b70-rope-audit-01"
