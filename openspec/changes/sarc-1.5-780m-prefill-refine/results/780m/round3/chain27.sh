#!/bin/bash
# chain27.sh (round 3, replacement build, owner decision 2026-10-08 22:55 UTC, option (a)): everything before the
# gate, on build head4. As chain25.sh, with the export taken from object stores only:
#   1. src/head4 was written by tools/export_recursive.sh c639d4760 (git archive of the commit; every submodule
#      and nested submodule fetched at its pinned commit into a bare repository under submodules/ and written
#      from there); here each exported submodule tree is checked against its commit (git diff-files after
#      read-tree) and compared with the directory head3 was built from
#   2. build head4 and head4-traced from that export (sarc/tools/build.sh, one hold unit each)
#   3. spirv_golden.py on head4; every .spv of head2 (candidate 11's gate build) and of head3 byte-compared
#   4. stage r3a-fused3sb-head4 and r3b-final-dev15-head4
#   5. dispatch smoke: SDPA kernel names with no environment, c10, c11, c11 + fused3 by variable, 780m-final,
#      and the three-kernel path (fused node off) under c11 and 780m-final
#   6. SDPA output of every correctness case (tiers all, extended, peaked, full, fused), fused3 against fused3sb,
#      byte for byte (and against head3's dumps), with the error against the fp64 reference printed for both
#   7. fused kernel time at a steady clock (40 warm-up + 10 timed runs), 3 runs per arm
# Resumes nothing: a rerun takes a new tag. No profiler variable anywhere (gl.sh refuses them).
A=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; ET=/home/doremy/hmz-sarc/executorch; T=$A/tools
A5=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-05; TAG=head4; W=raw/$TAG
env | grep -E '^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE|INTEL_MEASURE|MESA_GPU_TRACES)' && { echo "profiler variable set, refusing"; exit 97; }
export SARC_MOUNT_ROOT=/home/doremy/hmz-sarc ART780M=$A; cd $A
st() { echo "$(date -u +%FT%TZ) $*" >> logs/chain27.status; }
[[ -e build/$TAG ]] && { st "REFUSED: build/$TAG exists"; exit 2; }
S=$A/src/$TAG; C=$(cat $S/COMMIT) || exit 2
[[ $C == c639d47605ca1f2b5314e4b6afc6a58752386796 ]] || { st "REFUSED: export is of $C"; exit 2; }
st "start, commit $C"; mkdir -p $W

# 1. the export against its object stores, and against head3's directories
{ bad=0
  awk 'NR > 1 {print $1, $2, $4}' $S/MANIFEST | while read -r p pin store; do
    sha=${pin#pinned=}; B=${store#store=}
    GIT_INDEX_FILE=$B/check.index git --git-dir=$B --work-tree=$S/executorch/$p read-tree $sha
    GIT_INDEX_FILE=$B/check.index git --git-dir=$B --work-tree=$S/executorch/$p update-index -q --refresh > /dev/null
    # nested submodule directories are gitlinks in this index; diff-files does not descend into them
    if d=$(GIT_INDEX_FILE=$B/check.index git --git-dir=$B --work-tree=$S/executorch/$p diff-files --ignore-submodules=all --name-status) && [[ -z $d ]]; then
      echo "MATCHES-COMMIT $p $sha tree=$(git --git-dir=$B rev-parse $sha^{tree}) entries=$(GIT_INDEX_FILE=$B/check.index git --git-dir=$B ls-files | wc -l)"
    else echo "DIFFERS-FROM-COMMIT $p $sha"; echo "$d" | head -20; fi
    rm -f $B/check.index
  done
} > $W/export-check.txt 2>&1
[[ $(grep -c '^MATCHES-COMMIT' $W/export-check.txt) == $(( $(wc -l < $S/MANIFEST) - 1 )) ]] || { st EXPORT_CHECK_FAILED; exit 3; }
{ echo "# per submodule of $C: the object-store export of head4 (src/head4, nested submodules included) against the directory head3 was built from (src/head3, cp -a of the working copy); diff -r --no-dereference"
  git -C $ET ls-tree -r $C | awk '$2=="commit"{print $4}' | while read -r p; do
    d=$(diff -r --no-dereference -q src/head4/executorch/$p src/head3/executorch/$p 2>&1)
    n4=$(find src/head4/executorch/$p \( -type f -o -type l \) | wc -l); n3=$(find src/head3/executorch/$p \( -type f -o -type l \) | wc -l)
    if [[ -z $d ]]; then echo "IDENTICAL $p files head4=$n4 head3=$n3"
    else echo "DIFFERS $p files head4=$n4 head3=$n3: only-in-head4=$(grep -c '^Only in src/head4' <<< "$d") only-in-head3=$(grep -c '^Only in src/head3' <<< "$d") content=$(grep -c '^Files ' <<< "$d")"; echo "$d" | sed 's/^/    /' | head -40; fi
  done
  d=$(diff -r --no-dereference -q src/head4/executorch src/head3/executorch 2>&1); echo "whole tree: $(grep -c . <<< "$d") differing entries"; echo "$d" | sed 's/^/    /' | head -40
} > $W/submodules-head4-vs-head3.txt
st "export checked: $(grep -c '^MATCHES-COMMIT' $W/export-check.txt) trees match their commits; against head3: identical=$(grep -c '^IDENTICAL' $W/submodules-head4-vs-head3.txt) differs=$(grep -c '^DIFFERS' $W/submodules-head4-vs-head3.txt)"

# 2. build
{ date -u; echo $C; echo "source: $S/executorch (export from object stores only, MANIFEST beside it); working copy status at export:"; git -C $ET status --short; } > build/$TAG.src.txt 2>&1
$T/hold.sh run "build $TAG" $S/executorch/sarc/tools/build.sh --llama $S/executorch $A/build/$TAG > build/$TAG.log 2>&1; echo "rc=$? main" >> build/$TAG.src.txt
$T/hold.sh run "build $TAG-traced" $S/executorch/sarc/tools/build.sh --llama --traced --no-tests $S/executorch $A/build/$TAG-traced > build/$TAG-traced.log 2>&1; echo "rc=$? traced" >> build/$TAG.src.txt
date -u >> build/$TAG.src.txt; echo BUILD_BOTH_DONE >> build/$TAG.src.txt; cp $S/MANIFEST build/$TAG/EXPORT-MANIFEST
grep -q "rc=0 main" build/$TAG.src.txt && grep -q "rc=0 traced" build/$TAG.src.txt || { st BUILD_FAILED; exit 1; }
st "build done"

# 3. SPIR-V
mkdir -p $W/spirv
for d in llama backend; do
  N=$A/build/$TAG/$d/vulkan_compute_shaders
  python3 $ET/sarc/tools/spirv_golden.py $N $ET/sarc/golden/spirv.json > $W/spirv/golden-$d.txt 2>&1
  echo "rc=$?" >> $W/spirv/golden-$d.txt
  for old in head2:$A5 head3:$A; do o=${old%%:*}; O=${old#*:}/build/$o/$d/vulkan_compute_shaders
    { same=0; diff=0; miss=0
      for f in $O/*.spv; do b=$(basename $f)
        if [[ ! -f $N/$b ]]; then miss=$((miss + 1)); echo "MISSING $b"
        elif cmp -s $f $N/$b; then same=$((same + 1)); else diff=$((diff + 1)); echo "DIFFER $b"; fi
      done
      for f in $N/*.spv; do [[ -f $O/$(basename $f) ]] || echo "NEW $(basename $f) $(sha256sum < $f | cut -c1-64)"; done
      echo "$o $d: $(ls $O/*.spv | wc -l) spv; $TAG: $(ls $N/*.spv | wc -l) spv; identical=$same differ=$diff missing=$miss"
    } > $W/spirv/compare-$o-$d.txt
  done
done
st "spirv: $(tail -qn1 $W/spirv/golden-*.txt | tr '\n' ' ') $(tail -qn1 $W/spirv/compare-*.txt | tr '\n' ' ')"
for f in $W/spirv/compare-*.txt; do grep -q 'differ=0 missing=0' $f || { st SPIRV_CHECK_FAILED; exit 1; }; done
grep -q '^rc=0' $W/spirv/golden-llama.txt && grep -q '^rc=0' $W/spirv/golden-backend.txt || { st SPIRV_CHECK_FAILED; exit 1; }

# 4. stage
R3="ET_VK_SARC_DEV_PROFILE=780m-refine3"; C11="$R3 ET_VK_SARC_780M_PROFILE=c11"
F3="$C11 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko"
FIN="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final"
$T/stage.sh r3a-fused3sb-head4 $TAG "$F3" $TAG "$C11" "round 3 item A on head4: c11 with the fused3 pair by variable (parent) vs c11 as committed, fused3sb (candidate); same binary" > logs/stage-r3a-head4.out 2>&1
$T/stage.sh r3b-final-dev15-head4 $TAG "" $TAG "$FIN" "round 3 item B on head4: branch head with no environment (dispatches as dev/1.5) vs 780m-final; same binary" > logs/stage-r3b-head4.out 2>&1
MB=$A/stage/r3a-fused3sb-head4/test_llama_microbench
[[ -x $MB ]] || { st STAGE_FAILED; exit 1; }
{ for b in head3 head4; do sha256sum build/$b/llama/examples/models/llama/llama_main $(find build/$b/llama -name libllama_runner.so | head -1) build/$b/tests/test_llama_microbench; done; } > $W/binaries-sha256.txt
cool() { local t0=$SECONDS; while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 55000 && SECONDS - t0 < 300 )); do sleep 5; done; }

# 5. smoke
mkdir -p $W/smoke
sm() { local n=$1; shift; cool; env "$@" $T/gl.sh $MB --sdpa-correctness-only > $W/smoke/$n.log 2>&1; echo "rc=$?" >> $W/smoke/$n.log; }
sm none ETVK_DEVICE_INDEX=0
sm refine3 $R3
sm c10 $R3 ET_VK_SARC_780M_PROFILE=c10
sm c11 $C11
sm c11-fused3 $F3
sm final $FIN
sm final-no-unverified ET_VK_SARC_DEV_PROFILE=780m-final
sm final-with-c10 ET_VK_SARC_DEV_PROFILE=780m-final ET_VK_SARC_780M_PROFILE=c10
sm c11-nofused $C11 ET_VK_SARC_780M_SDPA_FUSED=
sm final-nofused $FIN ET_VK_SARC_780M_SDPA_FUSED=
st "smoke done"

# 6. bitwise, fused3 against fused3sb
mkdir -p $W/bitwise/dump-fused3 $W/bitwise/dump-fused3sb
for tier in all extended peaked full fused; do
  cool; env $F3 ET_VK_SDPA_ERROR_REPORT=1 ET_VK_DUMP_OUTPUT_DIR=$A/$W/bitwise/dump-fused3 $T/gl.sh $MB --sdpa-correctness-only --sdpa-tier=$tier > $W/bitwise/fused3-$tier.log 2>&1; echo "rc=$?" >> $W/bitwise/fused3-$tier.log
  cool; env $C11 ET_VK_SDPA_ERROR_REPORT=1 ET_VK_DUMP_OUTPUT_DIR=$A/$W/bitwise/dump-fused3sb $T/gl.sh $MB --sdpa-correctness-only --sdpa-tier=$tier > $W/bitwise/fused3sb-$tier.log 2>&1; echo "rc=$?" >> $W/bitwise/fused3sb-$tier.log
done
{ for f in $W/bitwise/dump-fused3/*.bin; do b=$(basename $f); g=$W/bitwise/dump-fused3sb/$b
    if [[ ! -f $g ]]; then echo "MISSING $b"; elif cmp -s $f $g; then echo "IDENTICAL $b $(stat -c %s $f) bytes sha256 $(sha256sum < $f | cut -c1-64)"; else echo "DIFFERENT $b"; fi
  done
  for g in $W/bitwise/dump-fused3sb/*.bin; do [[ -f $W/bitwise/dump-fused3/$(basename $g) ]] || echo "ONLY-IN-fused3sb $(basename $g)"; done
} > $W/bitwise/compare.txt
{ for a in fused3 fused3sb; do for f in $W/bitwise/dump-$a/*.bin; do b=$(basename $f)
    if cmp -s $f raw/bitwise/dump-$a/$b; then echo "IDENTICAL-TO-head3 $a $b"; else echo "DIFFERENT-FROM-head3 $a $b"; fi; done; done
} > $W/bitwise/compare-head3.txt
st "bitwise done: identical=$(grep -c '^IDENTICAL' $W/bitwise/compare.txt) other=$(grep -vc '^IDENTICAL' $W/bitwise/compare.txt); against head3's dumps identical=$(grep -c '^IDENTICAL' $W/bitwise/compare-head3.txt) other=$(grep -vc '^IDENTICAL' $W/bitwise/compare-head3.txt)"

# 7. kernel time at a steady clock
mkdir -p $W/kernel-time
for i in 1 2 3; do
  cool; env $F3 ET_VK_SDPA_PERF_RUNS=40,10 $T/gl.sh $MB --sdpa --json-out=$A/$W/kernel-time/fused3-r$i.json > $W/kernel-time/fused3-r$i.log 2>&1
  cool; env $C11 ET_VK_SDPA_PERF_RUNS=40,10 $T/gl.sh $MB --sdpa --json-out=$A/$W/kernel-time/fused3sb-r$i.json > $W/kernel-time/fused3sb-r$i.log 2>&1
done
st "DONE"
