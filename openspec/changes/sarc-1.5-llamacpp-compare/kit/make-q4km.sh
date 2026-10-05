#!/bin/bash
# make-q4km.sh <model dir> <llama-quantize> [threads]
# Q4_K_M from the F16 GGUF of each model, llama.cpp's DEFAULT quantization (no --pure, no forced tensor
# types): this arm is "what users run". Writes a note beside each file with the exact command and the
# tensor-type census llama-quantize prints, so the bits per weight can be stated with the result.
set -euo pipefail
D=$1; Q=$2; T=${3:-8}
commit=$(git -C "$(dirname "$Q")/../../llama.cpp" rev-parse HEAD 2>/dev/null || echo unknown)
for m in llama3_2_1b llama3_2_3b llama3_1_8b; do
  in=$D/${m}_f16.gguf; out=$D/${m}_q4_k_m.gguf; log=$D/${m}_q4_k_m.quantize.log
  [ -f "$in" ] || { echo "missing $in" >&2; exit 1; }
  [ -f "$out" ] && { echo "exists $out"; continue; }
  nice -n 10 "$Q" "$in" "$out.tmp" Q4_K_M "$T" > "$log" 2>&1
  mv "$out.tmp" "$out"
  {
    echo "gguf: $(basename "$out")"
    echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "intermediate: $(basename "$in") (August 2026 conversion, see ${m}_q4_0.gguf.txt for the source revision)"
    echo "llama.cpp: $commit"
    echo "command (exact): llama-quantize $(basename "$in") $(basename "$out") Q4_K_M $T"
    echo "size: $(stat -c %s "$out") bytes"
    echo "quantize summary:"; grep -E "model size|quant size|bpw|fallback" "$log" | sed 's/^/  /' || true
  } > "$out.txt"
  echo "done $out"
done
