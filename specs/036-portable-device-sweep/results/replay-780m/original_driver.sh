#!/usr/bin/env bash
# dev-igpu tile sweep: correctness gate + e2e ranking, one mode per invocation.
# usage: gate_and_rank.sh <4w|8da4w>
set -uo pipefail
MODE="$1"
WT=/home/doremy/sarc-acl/dev-igpu/executorch
OUT=/tmp/claude-1000/-home-doremy-sarc-acl-dev-executorch/e4a988b9-a0cc-4743-9bb1-e98f94402839/scratchpad/igpu_sweep
mkdir -p "$OUT"
BENCH=$WT/cmake-out-vk/backends/vulkan/test/custom_ops/test_coopmat_linear_bench
RUNNER=$WT/cmake-out-vk/examples/models/llama/llama_main
TOK=/home/doremy/checkpoints/llama3_2_1b/original/tokenizer.model
PROMPT=/home/doremy/checkpoints/llama3_2_1b/p2048_exact.txt

if [ "$MODE" = "4w" ]; then
  ENVVAR=ET_VK_Q4GSW_COOPMAT_VARIANT
  YAML=$WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.yaml
  KPAT="linear_q4gsw_coopmat"
  PTE=/home/doremy/checkpoints/llama3_2_1b/pte/llama3_2_1b_4w_buffer_ctx3072.pte
else
  ENVVAR=ET_VK_DQ8CA_COOPMAT_VARIANT
  YAML=$WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.yaml
  KPAT="linear_dq8ca_q4gsw_coopmat"
  PTE=/home/doremy/checkpoints/llama3_2_1b/pte/llama3_2_1b_8da4w_buffer_ctx3072.pte
fi

TOKENS=$(grep -o "tsweep_t[0-9]\+x[0-9]\+k[0-9]\+g[0-9][0-9]s[0-9]\+" "$YAML" | sort -u)
GATE_TSV=$OUT/${MODE}_gate.tsv
E2E_TSV=$OUT/${MODE}_e2e.tsv
: > "$GATE_TSV"; : > "$E2E_TSV"

# control first: default shipped dispatch, no env
for i in 1 2; do
  pf=$($RUNNER --model_path $PTE --tokenizer_path $TOK --prompt_file $PROMPT \
       --num_bos 1 --temperature 0 --max_new_tokens 1 --seq_len 3072 2>&1 \
       | grep -o '"prefill_token_per_sec":[0-9.]*' | cut -d: -f2)
  printf "CONTROL\t%s\n" "$pf" | tee -a "$E2E_TSV"
done

for t in $TOKENS; do
  # correctness gate
  log=$(env $ENVVAR=$t COOPMAT_BENCH_CORRECTNESS_ONLY=1 $BENCH 2>&1)
  fails=$(echo "$log" | grep -c "FAILED" || true)
  dispatched=$(echo "$log" | grep -c "${KPAT}_${t}" || true)
  if [ "$fails" -gt 0 ] || [ "$dispatched" -eq 0 ]; then
    printf "%s\tGATE-FAIL\tfails=%s dispatched=%s\n" "$t" "$fails" "$dispatched" | tee -a "$GATE_TSV"
    continue
  fi
  printf "%s\tPASS\n" "$t" | tee -a "$GATE_TSV"
  # e2e, 2 reps
  for i in 1 2; do
    pf=$(env $ENVVAR=$t $RUNNER --model_path $PTE --tokenizer_path $TOK --prompt_file $PROMPT \
         --num_bos 1 --temperature 0 --max_new_tokens 1 --seq_len 3072 2>&1 \
         | grep -o '"prefill_token_per_sec":[0-9.]*' | cut -d: -f2)
    printf "%s\t%s\n" "$t" "$pf" | tee -a "$E2E_TSV"
  done
done
echo "SWEEP-$MODE DONE"
