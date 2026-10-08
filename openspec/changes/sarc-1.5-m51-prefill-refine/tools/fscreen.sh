#!/bin/bash
# fscreen.sh <session> <out dir> <rounds> <env> : kernel-time screen of the M51 fused attention variants by ETDump.
#   <session> = a stage with an etdump candidate arm (stage/m51-LOCAL-ONLY/<session>/cand-etdump); <env> = its environment
#   (profile c1 or later). One warm ETDump per (round, variant): the 1B 4w cell for the head_dim 64 variants, the 3B 4w cell
#   for the head_dim 128 variants (the fused kernel is the same in 4w and 8da4w); the other head_dim keeps the profile's
#   variant. Variants are the two-pass `rk` forms of sarc_dev_m51_sdpa_fused3.yaml (the one-pass `rko` form gave wrong
#   rows on this driver). Result: <out>/round<r>/<cell>-<variant id>.etdp, then fscreen_summary.py applies the R8 margin
#   (3 % faster than the profile's variant in every round). Resumable (skips files already pulled); a board that is not fit
#   stops it; no timing figure of a trace is a tok/s. One coordinator-hold unit.
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "fused screen fscreen.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/dev.sh"
SES=$1; O=$2; ROUNDS=$3; BENV=$4
S=$ART/stage/$LOC/$SES; DS=$DEV_ROOT/stage/$SES/cand-etdump; mkdir -p "$O"
D64=(t32x32g11s32rk t16x32g11s32rk t32x64g11s32rk t32x32g11s64rk t32x64g11s64rk t16x64g11s64rk)
D128=(t16x64g11s32rk t16x32g11s32rk t32x32g11s32rk t16x64g11s64rk t32x32g11s64rk t32x64g11s64rk)
DEF64=${D64[0]}; DEF128=${D128[0]}
IDLE=$(gtemp)
for r in $(seq 1 "$ROUNDS"); do mkdir -p "$O/round$r"
  for hd in 64 128; do
    if [[ $hd == 64 ]]; then V=("${D64[@]}"); cell=1b-4w; model=llama3_2_1b; else V=("${D128[@]}"); cell=3b-4w; model=llama3_2_3b; fi
    for v in "${V[@]}"; do
      if [[ $hd == 64 ]]; then list="fused3_d64_$v,fused3_d128_$DEF128"; else list="fused3_d64_$DEF64,fused3_d128_$v"; fi
      id=d${hd}_$v; f=$O/round$r/$cell-$id
      [[ -s $f.etdp ]] && continue
      t0=$SECONDS; best=999; tb=$SECONDS
      while :; do t=$(gtemp); [[ $t =~ ^[0-9]+$ ]] || break; (( t < best )) && { best=$t; tb=$SECONDS; }
        (( t <= IDLE + 5 || SECONDS - tb >= 30 || SECONDS - t0 >= 300 )) && break; sleep 5; done
      st=$(device_state); [[ $st == ok ]] || { echo "board not fit before $f: $st"; [[ $st == device_gone ]] && echo "board gone before fscreen $f" >> "$ART/ABORTED"; exit 3; }
      A shell "cd $DS && rm -f fs.etdp && $BENV ET_VK_SARC_M51_SDPA_FUSED=$list LD_LIBRARY_PATH=$DS timeout 1190 ./llama_main --model_path=$DEV_ROOT/models/${model}_4w_embq_ctx3072.pte --tokenizer_path=$DEV_ROOT/models/tokenizer.model --prompt_file=prompt_2048.txt --max_new_tokens=1 --temperature=0 --warmup --etdump_path=fs.etdp < /dev/null > fs.log 2>&1; echo RC=\$? >> fs.log" < /dev/null > /dev/null 2>&1
      alive || { echo "board gone during fscreen $f $(date -u +%FT%TZ)" | tee -a "$ART/ABORTED"; exit 3; }
      A pull "$DS/fs.log" "$f.log" > /dev/null; A pull "$DS/fs.etdp" "$f.etdp" > /dev/null 2>&1
      echo "fscreen r$r $cell $id $(tail -1 "$f.log") $(date -u +%FT%TZ)"
    done
  done
  "$ART/venv/m51/bin/python" -I "$TOOLS/trace_families.py" "$O/round$r" > "$O/round$r/families.csv" 2> "$O/round$r/families.err"
done
python3 -I "$TOOLS/fscreen_summary.py" "$O" "$ROUNDS" > "$O/summary.csv" 2>&1; tail -20 "$O/summary.csv"
echo FSCREEN_DONE
