#!/bin/bash
# test_nexttoken.sh: CPU-only regression of the parent-vs-candidate next-token comparison of e2e5.sh and of the
# prompt identities gate_check.py requires. llama_main is a shell stub that prints a token and an observer
# line (no model, no GPU work); artifact directory and HOME (the lock) are temporary. One cell (1b 4w), one
# repeat. Cases: identical arms -> SAME on prompt_2048.txt, prompt_check.txt and r1304.txt; a candidate that
# differs only on the unaligned prompt -> DIFFER there, session not OK, gate_check FAIL for that cell; no
# unaligned prompt staged -> INVALID, session not OK; a candidate that crashes on the unaligned prompt ->
# INVALID (two empty outputs are never SAME). About 4.5 minutes (the session's 60 s idle wait, four times).
set -u
T=$(dirname "$(readlink -f "$0")"); W=$(mktemp -d); fails=0
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1   [$2]"; fails=$((fails + 1)); fi; }
unset B580_TOP; export HOME=$W/home B580_ARTIFACTS=$W/art; mkdir -p $HOME/.cache/gpu-lab $B580_ARTIFACTS
stage() { # stage <session> <cand env line or ""> <with unaligned prompt: 1|0>
  local D=$B580_ARTIFACTS/stage/$1 d; mkdir -p $D/{parent,cand}
  cat > $D/parent/llama_main <<'S'
#!/bin/bash
# stub runner: prints a "next token" that depends only on the prompt (and on STUB_* from the arm's env file)
while [[ $# -gt 0 ]]; do [[ $1 == --prompt_file ]] && P=$(basename $2); shift; done
case $P in prompt_2048.txt) n=2048 ;; prompt_check.txt) n=1972 ;; *) n=1792 ;; esac
[[ $n == 1792 && -n ${STUB_UNALIGNED_CRASH:-} ]] && exit 3
a=$(date +%s%3N); sleep 0.4; b=$(date +%s%3N); tok="token-for-$P"; [[ $n == 1792 ]] && tok+=${STUB_UNALIGNED_SUFFIX:-}
echo "$tok"
echo "PyTorchObserver {\"prompt_tokens\":$n,\"generated_tokens\":0,\"prefill_token_per_sec\":1000.0,\"inference_start_ms\":$a,\"prompt_eval_end_ms\":$b,\"model_execution_start_ms\":$a,\"model_execution_end_ms\":$b}"
S
  chmod +x $D/parent/llama_main; cp $D/parent/llama_main $D/cand/; for d in parent cand; do : > $D/$d/libllama_runner.so; : > $D/$d/env; done
  [[ -n $2 ]] && echo "$2" > $D/cand/env
  : > $D/prompt_2048.txt; : > $D/prompt_check.txt; [[ $3 == 1 ]] && : > $D/r1304.txt
  echo test > $D/STAGE.md; }
session() { bash $T/e2e5.sh --stage $B580_ARTIFACTS/stage/$1 --out raw --lock test --calibrate --reps 1 --extra 0 --models 1b --schemes 4w > $W/$1.out 2>&1; RC=$?; D=$B580_ARTIFACTS/stage/$1; }
cell() { python3 $T/gate_check.py $D --parent /nonexistent | grep "next token parent vs cand: 1b 4w"; }

stage same "" 1; session same
ok "identical arms: SAME on the three named prompts" 'grep -qx "1b,4w,prompt_2048.txt:SAME,prompt_check.txt:SAME,r1304.txt:SAME" $D/raw/nexttoken.csv'
ok "six runs: 2 timed, 2 real-text, 2 unaligned (1792 tokens)" '[[ $(grep -c "logs/unaligned-1b-4w-.*,1792,0," $D/raw/runs.csv) == 2 && $(wc -l < $D/raw/runs.csv) == 7 ]]'
ok "checker accepts the cell" 'cell | grep -q "^PASS"'
ok "stub clock is not a calibration (session not OK, nothing stored)" '[[ $RC == 1 && ! -s $B580_ARTIFACTS/clkmin_mhz ]] || [[ $RC == 0 ]]'

stage differ "STUB_UNALIGNED_SUFFIX=-other" 1; session differ
ok "candidate differs only on the unaligned prompt: DIFFER there" 'grep -qx "1b,4w,prompt_2048.txt:SAME,prompt_check.txt:SAME,r1304.txt:DIFFER" $D/raw/nexttoken.csv'
ok "session not OK, reason recorded" '[[ $RC == 1 ]] && grep -q "E2E5_INCOMPLETE.*1b-4w:nexttoken_SAME_SAME_DIFFER" $D/raw/done.txt'
ok "checker rejects the cell" 'cell | grep -q "^FAIL.*r1304.txt:DIFFER"'

stage crash "STUB_UNALIGNED_CRASH=1" 1; session crash
ok "candidate fails on the unaligned prompt: INVALID, never SAME" 'grep -qx "1b,4w,prompt_2048.txt:SAME,prompt_check.txt:SAME,r1304.txt:INVALID" $D/raw/nexttoken.csv && [[ $RC == 1 ]]'
ok "checker rejects the cell" 'cell | grep -q "^FAIL.*r1304.txt:INVALID"'

stage nounal "" 0; session nounal
ok "no unaligned prompt staged: INVALID, session not OK" 'grep -q "^1b,4w,prompt_2048.txt:SAME,prompt_check.txt:SAME,unaligned_prompt_missing:INVALID$" $D/raw/nexttoken.csv && [[ $RC == 1 ]]'
ok "checker rejects the cell" 'cell | grep -q "^FAIL.*no unaligned prompt"'
[[ $fails == 0 ]] && { echo TEST_NEXTTOKEN_PASS; rm -rf $W; exit 0; }
echo "TEST_NEXTTOKEN_FAIL ($fails), evidence kept in $W"; exit 1
