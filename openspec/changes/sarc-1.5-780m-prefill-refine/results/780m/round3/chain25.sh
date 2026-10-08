#!/bin/bash
# chain25.sh <commit> (round 3, 2026-10-08): everything before the gate.
#   1. export of one commit: git archive of <commit> into src/head3/executorch; the submodule directories are
#      copied from the working copy (this clone has no submodule object stores, .git/modules is empty; every
#      earlier build of this campaign used the same directories), with the pinned commit and a content hash each
#      in src/head3/MANIFEST
#   2. build head3 and head3-traced from that export (sarc/tools/build.sh, one hold unit each)
#   3. spirv_golden.py on head3; every .spv of head2 (candidate 11's gate build) byte-compared with head3
#   4. stage r3a-fused3sb and r3b-final-dev15
#   5. dispatch smoke: SDPA kernel names with no environment, c10, c11, c11 + fused3 by variable, 780m-final
#   6. SDPA output of every correctness case (tiers all, extended, peaked, full, fused), fused3 against fused3sb,
#      byte for byte, with the error against the fp64 reference printed for both
#   7. fused kernel time at a steady clock (40 warm-up + 10 timed runs), 3 runs per arm
# Resumes nothing: a rerun takes a new tag. No profiler variable anywhere (gl.sh refuses them).
A=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; ET=/home/doremy/hmz-sarc/executorch; T=$A/tools
A5=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-05; TAG=head3; C=$1
env | grep -E '^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE|INTEL_MEASURE|MESA_GPU_TRACES)' && { echo "profiler variable set, refusing"; exit 97; }
export SARC_MOUNT_ROOT=/home/doremy/hmz-sarc ART780M=$A; cd $A
st() { echo "$(date -u +%FT%TZ) $*" >> logs/chain25.status; }
[[ -e src/$TAG || -e build/$TAG ]] && { st "REFUSED: $TAG exists"; exit 2; }
C=$(git -C $ET rev-parse "$C^{commit}") || exit 2
st "start, commit $C"

# 1. export
S=$A/src/$TAG; mkdir -p $S/executorch
git -C $ET archive $C | tar -x -C $S/executorch || { st EXPORT_FAILED; exit 3; }
echo ". $C git-archive" > $S/MANIFEST
git -C $ET ls-tree -r $C | awk '$2=="commit"{print $3, $4}' | while read -r sha path; do
  cp -a $ET/$path/. $S/executorch/$path/ || exit 3
  h=$(cd $S/executorch/$path && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)
  echo "$path pinned=$sha source=working-copy-directory files_sha256=$h" >> $S/MANIFEST
done
echo $C > $S/COMMIT; st "export done, $(wc -l < $S/MANIFEST) manifest entries"

# 2. build
{ date -u; echo $C; echo "source: $S/executorch (export, MANIFEST beside it); working copy status at export:"; git -C $ET status --short; } > build/$TAG.src.txt 2>&1
$T/hold.sh run "build $TAG" $S/executorch/sarc/tools/build.sh --llama $S/executorch $A/build/$TAG > build/$TAG.log 2>&1; echo "rc=$? main" >> build/$TAG.src.txt
$T/hold.sh run "build $TAG-traced" $S/executorch/sarc/tools/build.sh --llama --traced --no-tests $S/executorch $A/build/$TAG-traced > build/$TAG-traced.log 2>&1; echo "rc=$? traced" >> build/$TAG.src.txt
date -u >> build/$TAG.src.txt; echo BUILD_BOTH_DONE >> build/$TAG.src.txt; cp $S/MANIFEST build/$TAG/EXPORT-MANIFEST
grep -q "rc=0 main" build/$TAG.src.txt && grep -q "rc=0 traced" build/$TAG.src.txt || { st BUILD_FAILED; exit 1; }
st "build done"

# 3. SPIR-V
mkdir -p raw/spirv
for d in llama backend; do
  python3 $ET/sarc/tools/spirv_golden.py $A/build/$TAG/$d/vulkan_compute_shaders $ET/sarc/golden/spirv.json > raw/spirv/golden-$d.txt 2>&1
  echo "rc=$?" >> raw/spirv/golden-$d.txt
  O=$A5/build/head2/$d/vulkan_compute_shaders; N=$A/build/$TAG/$d/vulkan_compute_shaders
  { same=0; diff=0; miss=0
    for f in $O/*.spv; do b=$(basename $f)
      if [[ ! -f $N/$b ]]; then miss=$((miss + 1)); echo "MISSING $b"
      elif cmp -s $f $N/$b; then same=$((same + 1)); else diff=$((diff + 1)); echo "DIFFER $b"; fi
    done
    for f in $N/*.spv; do [[ -f $O/$(basename $f) ]] || echo "NEW $(basename $f) $(sha256sum < $f | cut -c1-64)"; done
    echo "head2 $d: $(ls $O/*.spv | wc -l) spv; $TAG: $(ls $N/*.spv | wc -l) spv; identical=$same differ=$diff missing=$miss"
  } > raw/spirv/compare-head2-$d.txt
done
st "spirv: $(tail -qn1 raw/spirv/golden-*.txt | tr '\n' ' ') $(tail -qn1 raw/spirv/compare-head2-*.txt | tr '\n' ' ')"
grep -q 'differ=0 missing=0' raw/spirv/compare-head2-llama.txt && grep -q 'differ=0 missing=0' raw/spirv/compare-head2-backend.txt \
  && grep -q '^rc=0' raw/spirv/golden-llama.txt && grep -q '^rc=0' raw/spirv/golden-backend.txt || { st SPIRV_CHECK_FAILED; exit 1; }

# 4. stage
R3="ET_VK_SARC_DEV_PROFILE=780m-refine3"; C11="$R3 ET_VK_SARC_780M_PROFILE=c11"
F3="$C11 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko"
FIN="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final"
$T/stage.sh r3a-fused3sb $TAG "$F3" $TAG "$C11" "round 3 item A: c11 with the fused3 pair by variable (parent) vs c11 as committed, fused3sb (candidate); same binary" > logs/stage-r3a.out 2>&1
$T/stage.sh r3b-final-dev15 $TAG "" $TAG "$FIN" "round 3 item B: branch head with no environment (dispatches as dev/1.5) vs 780m-final; same binary" > logs/stage-r3b.out 2>&1
MB=$A/stage/r3a-fused3sb/test_llama_microbench
cool() { local t0=$SECONDS; while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 55000 && SECONDS - t0 < 300 )); do sleep 5; done; }

# 5. smoke
mkdir -p raw/smoke
sm() { local n=$1; shift; cool; env "$@" $T/gl.sh $MB --sdpa-correctness-only > raw/smoke/$n.log 2>&1; echo "rc=$?" >> raw/smoke/$n.log; }
sm none ETVK_DEVICE_INDEX=0
sm refine3 $R3
sm c10 $R3 ET_VK_SARC_780M_PROFILE=c10
sm c11 $C11
sm c11-fused3 $F3
sm final $FIN
sm final-no-unverified ET_VK_SARC_DEV_PROFILE=780m-final
sm final-with-c10 ET_VK_SARC_DEV_PROFILE=780m-final ET_VK_SARC_780M_PROFILE=c10
st "smoke done"

# 6. bitwise, fused3 against fused3sb
mkdir -p raw/bitwise/dump-fused3 raw/bitwise/dump-fused3sb
for tier in all extended peaked full fused; do
  cool; env $F3 ET_VK_SDPA_ERROR_REPORT=1 ET_VK_DUMP_OUTPUT_DIR=$A/raw/bitwise/dump-fused3 $T/gl.sh $MB --sdpa-correctness-only --sdpa-tier=$tier > raw/bitwise/fused3-$tier.log 2>&1; echo "rc=$?" >> raw/bitwise/fused3-$tier.log
  cool; env $C11 ET_VK_SDPA_ERROR_REPORT=1 ET_VK_DUMP_OUTPUT_DIR=$A/raw/bitwise/dump-fused3sb $T/gl.sh $MB --sdpa-correctness-only --sdpa-tier=$tier > raw/bitwise/fused3sb-$tier.log 2>&1; echo "rc=$?" >> raw/bitwise/fused3sb-$tier.log
done
{ for f in raw/bitwise/dump-fused3/*.bin; do b=$(basename $f); g=raw/bitwise/dump-fused3sb/$b
    if [[ ! -f $g ]]; then echo "MISSING $b"; elif cmp -s $f $g; then echo "IDENTICAL $b $(stat -c %s $f) bytes sha256 $(sha256sum < $f | cut -c1-64)"; else echo "DIFFERENT $b"; fi
  done
  for g in raw/bitwise/dump-fused3sb/*.bin; do [[ -f raw/bitwise/dump-fused3/$(basename $g) ]] || echo "ONLY-IN-fused3sb $(basename $g)"; done
} > raw/bitwise/compare.txt
st "bitwise done: identical=$(grep -c '^IDENTICAL' raw/bitwise/compare.txt) other=$(grep -vc '^IDENTICAL' raw/bitwise/compare.txt)"

# 7. kernel time at a steady clock
mkdir -p raw/kernel-time
for i in 1 2 3; do
  cool; env $F3 ET_VK_SDPA_PERF_RUNS=40,10 $T/gl.sh $MB --sdpa --json-out=$A/raw/kernel-time/fused3-r$i.json > raw/kernel-time/fused3-r$i.log 2>&1
  cool; env $C11 ET_VK_SDPA_PERF_RUNS=40,10 $T/gl.sh $MB --sdpa --json-out=$A/raw/kernel-time/fused3sb-r$i.json > raw/kernel-time/fused3sb-r$i.log 2>&1
done
st "DONE"
