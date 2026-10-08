#!/bin/bash
# test_resume.sh: regression of the resumable screens (screen.sh, screen_sdpa.sh, screen_rows.py) on COPIES of
# existing raw results (raw/screen5-4w and raw/screen1-sdpa of the real artifact directory). No kernel is
# measured: the artifact directory is a scratch copy and test_llama_microbench is a stub that records every
# launch and replays a saved result, so "GPU work was rerun" is a countable fact. Cases, for both screens:
#   empty     the CSV is absent                      -> every row recovered from the cached results, 0 launches
#   partial   the CSV is cut in the middle of a line -> the missing rows recovered, 0 launches, damaged CSV kept
#   complete  the CSV is complete                    -> 0 launches, CSV byte-identical
#   marker    (SDPA) the status marker exists and the rows do not: the old script skipped such a run forever
#   interrupt one run's result is missing / marked interrupted, with partial rows -> exactly 1 launch, the
#             earlier attempt and its rows preserved under superseded/, final CSV complete
#   incomplete evidence (screen_rows.py alone, in an otherwise empty directory): one saved row and nothing else,
#             a cached JSON with one case, a cached SDPA log with one RESULT line -> rerun requested, the
#             attempt preserved; a CSV cut inside its last field (field count still fits) -> that record is
#             dropped and recovered from the cache
# After every case: exactly one row per (configuration, repeat, shape) and the original logs / JSON unchanged.
set -u
T=$(dirname "$(readlink -f "$0")"); REAL=${B580_ARTIFACTS:-$(cd $T/../../../../.. && pwd)/.artifacts}
W=$(mktemp -d -p $REAL/tmp resume-test.XXXXXX); fails=0
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1   [$2]"; fails=$((fails + 1)); fi; }
mkdir -p $W/raw $W/build/stub/tests && cp -r $REAL/raw/screen5-4w $W/raw/lin && cp -r $REAL/raw/screen1-sdpa $W/raw/sdpa || exit 2
rm -f $W/raw/lin/rows.csv $W/raw/lin/summary.csv $W/raw/sdpa/summary.csv; rm -rf $W/raw/*/superseded
cp $W/raw/sdpa/screen.csv $W/sdpa-original.csv; cp $W/raw/lin/4w-base-r1.json $W/replay.json; cp $W/raw/sdpa/b580-refine0-r1.log $W/replay.log
cat > $W/build/stub/tests/test_llama_microbench <<S
#!/bin/bash
# stub: records the launch, replays a saved result
echo "\$*" >> $W/launches.txt
for a in "\$@"; do [[ \$a == --json-out=* ]] && cp $W/replay.json "\${a#--json-out=}"; done
[[ " \$* " == *" --sdpa "* ]] && cat $W/replay.log
exit 0
S
chmod +x $W/build/stub/tests/test_llama_microbench; : > $W/launches.txt
export B580_ARTIFACTS=$W; unset B580_TOP B580_SCREEN_RECOVER_ONLY
LT=$(cd $W/raw/lin && ls 4w-*-r1.json | sed 's/^4w-//; s/-r1\.json$//' | tr '\n' ' '); SP=$(cd $W/raw/sdpa && ls *-r1.log | sed 's/-r1\.log$//' | tr '\n' ' ')
manifest() { (cd $W/raw/$1 && find . -maxdepth 1 -type f \( -name '*.json' -o -name '*.log' -o -name '*.rc' \) | sort | xargs sha256sum); }
manifest lin > $W/lin.sha; manifest sdpa > $W/sdpa.sha
nl=$(echo $LT | wc -w); np=$(echo $SP | wc -w); WANT_L=$((nl * 3 * 12)); WANT_S=$((np * 2 * 24))
uniq_rows() { python3 - $1 $2 <<'PY'
import csv, sys
r = list(csv.reader(open(sys.argv[1])))[1:]; k = [tuple(x[:int(sys.argv[2])]) for x in r]
print(len(r) if len(set(k)) == len(k) and all(len(x) == len(r[0]) for x in r) else -1)
PY
}
lin() { $T/screen.sh lin stub 4w 3 $LT > $W/out.txt 2>&1; echo $?; }
sdpa() { $T/screen_sdpa.sh sdpa stub 2 $SP > $W/out.txt 2>&1; echo $?; }
same() { manifest $1 | diff -q - $W/$1.sha > /dev/null; }
launches() { grep -c . $W/launches.txt; }

echo "== linear ($nl tokens x 3 rounds x 12 shapes = $WANT_L rows)"
ok "empty: status 0" '[[ $(lin) == 0 ]]'
ok "empty: all rows recovered, one per key" '[[ $(uniq_rows $W/raw/lin/rows.csv 6) == $WANT_L ]]'
ok "empty: no launch, logs and JSON unchanged" '[[ $(launches) == 0 ]] && same lin'
cp $W/raw/lin/rows.csv $W/lin-complete.csv
head -c $(( $(wc -c < $W/lin-complete.csv) / 2 )) $W/lin-complete.csv > $W/raw/lin/rows.csv
ok "partial: status 0" '[[ $(lin) == 0 ]]'
ok "partial: all rows again, one per key" '[[ $(uniq_rows $W/raw/lin/rows.csv 6) == $WANT_L ]]'
ok "partial: same rows as the complete CSV" 'diff -q <(sort $W/raw/lin/rows.csv) <(sort $W/lin-complete.csv) > /dev/null'
ok "partial: damaged CSV preserved" 'ls $W/raw/lin/superseded/csv-damaged-*/rows.csv > /dev/null 2>&1'
ok "partial: no launch, logs and JSON unchanged" '[[ $(launches) == 0 ]] && same lin'
cp $W/raw/lin/rows.csv $W/lin-before.csv; e0=$(wc -l < $W/raw/lin/env.txt)
ok "complete: status 0, CSV byte-identical, no launch" '[[ $(lin) == 0 ]] && cmp -s $W/raw/lin/rows.csv $W/lin-before.csv && [[ $(launches) == 0 ]] && same lin'
ok "env.txt appended to, not overwritten" '[[ $(wc -l < $W/raw/lin/env.txt) -gt $e0 ]] && grep -q "$(head -1 $REAL/raw/screen5-4w/env.txt | cut -c1-64)" $W/raw/lin/env.txt'
# an interrupted attempt: marker without status, no JSON, a stale log, and 5 of its 12 rows in the CSV
t1=$(echo $LT | cut -d' ' -f2); S=$W/raw/lin/4w-$t1-r2; rm -f $S.json; echo "stale log of the interrupted attempt" > $S.log; date > $S.started
python3 - $W/raw/lin/rows.csv $t1 <<'PY'
import sys
L = open(sys.argv[1]).read().splitlines(True); n = 0; out = []
for l in L:
    if l.startswith(sys.argv[2] + ",2,"):
        n += 1
        if n > 5: continue
    out.append(l)
open(sys.argv[1], "w").writelines(out)
PY
ok "interrupt: status 0" '[[ $(lin) == 0 ]]'
ok "interrupt: exactly one launch, of that token" '[[ $(launches) == 1 ]] && grep -q "r2.json" $W/launches.txt'
ok "interrupt: stale log, marker and partial rows preserved" 'd=$(ls -d $W/raw/lin/superseded/interrupted-4w-$t1-r2-*) && grep -q "stale log" $d/4w-$t1-r2.log && [[ -f $d/4w-$t1-r2.started ]] && [[ $(grep -c "^$t1,2," $d/rows.csv) == 5 ]]'
ok "interrupt: complete again, one row per key" '[[ $(uniq_rows $W/raw/lin/rows.csv 6) == $WANT_L ]]'
ok "interrupt: every other log and JSON unchanged" 'manifest lin | grep -v "4w-$t1-r2\." | diff -q - <(grep -v "4w-$t1-r2\." $W/lin.sha) > /dev/null'

echo "== SDPA ($np profiles x 2 rounds x 24 rows = $WANT_S rows)"; : > $W/launches.txt
nt() { cut -d, -f1-8,10 $1 | sort; }   # all columns but temp_c, which a status marker saved before this fix does not hold
rm -f $W/raw/sdpa/screen.csv
ok "empty / marker: status 0 although every status marker exists" '[[ $(sdpa) == 0 ]]'
ok "empty: all rows recovered, one per key" '[[ $(uniq_rows $W/raw/sdpa/screen.csv 5) == $WANT_S ]]'
ok "empty: recovered rows equal the original CSV (temperature column aside)" 'diff -q <(nt $W/raw/sdpa/screen.csv) <(nt $W/sdpa-original.csv) > /dev/null'
ok "empty: no launch, logs unchanged" '[[ $(launches) == 0 ]] && same sdpa'
head -c $(( $(wc -c < $W/sdpa-original.csv) / 3 )) $W/sdpa-original.csv > $W/raw/sdpa/screen.csv
ok "partial: status 0" '[[ $(sdpa) == 0 ]]'
ok "partial: all rows, one per key, original rows kept as they were" '[[ $(uniq_rows $W/raw/sdpa/screen.csv 5) == $WANT_S ]] && diff -q <(nt $W/raw/sdpa/screen.csv) <(nt $W/sdpa-original.csv) > /dev/null && [[ $(grep -c -x -F -f <(head -500 $W/sdpa-original.csv) $W/raw/sdpa/screen.csv) == 500 ]]'
ok "partial: damaged CSV preserved, no launch, logs unchanged" 'ls $W/raw/sdpa/superseded/csv-damaged-*/screen.csv > /dev/null 2>&1 && [[ $(launches) == 0 ]] && same sdpa'
cp $W/sdpa-original.csv $W/raw/sdpa/screen.csv
ok "complete: status 0, CSV byte-identical, no launch" '[[ $(sdpa) == 0 ]] && cmp -s $W/raw/sdpa/screen.csv $W/sdpa-original.csv && [[ $(launches) == 0 ]] && same sdpa'
p1=$(echo $SP | cut -d' ' -f3); L=$W/raw/sdpa/$p1-r2.log; rm -f $L.rc; echo "stale log of the interrupted attempt" > $L
grep -v "^$p1,2,llama-3.1-8b," $W/sdpa-original.csv > $W/raw/sdpa/screen.csv
ok "interrupt: status 0, exactly one launch" '[[ $(sdpa) == 0 ]] && [[ $(launches) == 1 ]]'
ok "interrupt: stale log and partial rows preserved" 'd=$(ls -d $W/raw/sdpa/superseded/interrupted-$p1-r2-*) && grep -q "stale log" $d/$p1-r2.log && [[ $(grep -c "^$p1,2," $d/screen.csv) == 16 ]]'
ok "interrupt: complete again, one row per key" '[[ $(uniq_rows $W/raw/sdpa/screen.csv 5) == $WANT_S ]]'
ok "interrupt: every other log unchanged" 'manifest sdpa | grep -v "/$p1-r2\.log" | diff -q - <(grep -v "/$p1-r2\.log" $W/sdpa.sha) > /dev/null'
echo "== incomplete evidence is never taken for a complete run (review follow-up 2)"; : > $W/launches.txt
RL=$T/screen_rows.py; H_L=$(head -1 $W/lin-complete.csv); H_S=$(head -1 $W/sdpa-original.csv)
fresh() { rm -rf $W/x; mkdir -p $W/x; }
fresh; { echo "$H_L"; grep -m1 '^base,1,' $W/lin-complete.csv; } > $W/x/rows.csv
ok "linear: 1 of 12 rows, nothing else in the directory -> rerun requested (--no-supersede)" 'python3 $RL linear $W/x 4w base 1 --no-supersede > /dev/null; [[ $? == 10 ]] && [[ $(wc -l < $W/x/rows.csv) == 2 ]]'
ok "linear: the same without the flag -> rerun, partial row preserved and removed from the live CSV" 'python3 $RL linear $W/x 4w base 1 > /dev/null; [[ $? == 10 ]] && [[ $(wc -l < $W/x/rows.csv) == 1 ]] && [[ $(grep -c "^base,1," $W/x/superseded/interrupted-4w-base-r1-*/rows.csv) == 1 ]]'
fresh; { echo "$H_S"; grep -m1 '^base,1,' $W/sdpa-original.csv; } > $W/x/screen.csv
ok "SDPA: 1 of 24 rows, nothing else -> rerun, partial row preserved" 'python3 $RL sdpa $W/x base 1 --no-supersede > /dev/null; [[ $? == 10 ]] && python3 $RL sdpa $W/x base 1 > /dev/null; [[ $? == 10 ]] && [[ $(wc -l < $W/x/screen.csv) == 1 ]] && [[ $(grep -c "^base,1," $W/x/superseded/interrupted-base-r1-*/screen.csv) == 1 ]]'
fresh; python3 -c "
import json,sys; j=json.load(open(sys.argv[1])); j['cases']=j['cases'][:1]; json.dump(j,open(sys.argv[2],'w'))" $W/replay.json $W/x/4w-base-r1.json; echo "0 42" > $W/x/4w-base-r1.rc
ok "linear: a cached JSON with 1 of 12 cases is not a result -> rerun, nothing recovered, JSON preserved" 'python3 $RL linear $W/x 4w base 1 > /dev/null; [[ $? == 10 ]] && [[ $(wc -l < $W/x/rows.csv) == 1 ]] && ls $W/x/superseded/interrupted-4w-base-r1-*/4w-base-r1.json > /dev/null'
fresh; grep -m1 '^RESULT,sdpa,.*,prefill,' $W/replay.log > $W/x/p-r1.log; echo "0 42" > $W/x/p-r1.log.rc
ok "SDPA: a cached log with 1 RESULT line is not a result -> rerun, nothing recovered, log preserved" 'python3 $RL sdpa $W/x p 1 > /dev/null; [[ $? == 10 ]] && [[ $(wc -l < $W/x/screen.csv) == 1 ]] && ls $W/x/superseded/interrupted-p-r1-*/p-r1.log > /dev/null'
fresh; cp $W/replay.json $W/x/4w-base-r1.json; echo "0 42" > $W/x/4w-base-r1.rc
ok "linear: a complete cache is still recovered (12 rows, temperature 42)" 'python3 $RL linear $W/x 4w base 1 > /dev/null; [[ $? == 0 ]] && [[ $(grep -c ",0,42$" $W/x/rows.csv) == 12 ]]'
cp $W/x/rows.csv $W/x-complete.csv; head -c -2 $W/x-complete.csv > $W/x/rows.csv
ok "truncation inside the last field: the cut record still has 16 fields and reads temperature 4" '[[ $(tail -c 4 $W/x/rows.csv) == ",0,4" ]] && [[ $(tail -1 $W/x/rows.csv | tr -cd , | wc -c) == 15 ]]'
ok "truncation inside the last field: record dropped and recovered from the cache as 42, damaged CSV preserved" 'python3 $RL linear $W/x 4w base 1 > /dev/null; [[ $? == 0 ]] && diff -q <(sort $W/x/rows.csv) <(sort $W/x-complete.csv) > /dev/null && [[ $(tail -c 4 $W/x/superseded/csv-damaged-*/rows.csv) == ",0,4" ]]'
fresh; cp $W/replay.log $W/x/p-r1.log; echo "0 42" > $W/x/p-r1.log.rc; python3 $RL sdpa $W/x p 1 > /dev/null; cp $W/x/screen.csv $W/x-complete.csv; head -c -2 $W/x-complete.csv > $W/x/screen.csv
ok "SDPA: truncation inside the last field recovered the same way" 'python3 $RL sdpa $W/x p 1 > /dev/null; [[ $? == 0 ]] && [[ $(wc -l < $W/x-complete.csv) == 25 ]] && diff -q <(sort $W/x/screen.csv) <(sort $W/x-complete.csv) > /dev/null && ls $W/x/superseded/csv-damaged-*/screen.csv > /dev/null'
ok "none of these cases launched anything" '[[ $(launches) == 0 ]]'
echo "== recover-only mode launches nothing"; : > $W/launches.txt; rm -f $W/raw/lin/4w-base-r3.json; sed -i '/^base,3,/d' $W/raw/lin/rows.csv
ok "recover-only: reports the missing run, launches nothing, moves nothing" '[[ $(B580_SCREEN_RECOVER_ONLY=1 lin) == 0 ]] && grep -q "WOULD_RUN 4w base r3" $W/out.txt && [[ $(launches) == 0 ]] && [[ -f $W/raw/lin/4w-base-r3.log ]]'
echo "scratch: $W"; [[ $fails == 0 ]] && { echo "RESUME_TEST_OK"; rm -rf $W; exit 0; }; echo "RESUME_TEST_FAILED ($fails)"; exit 1
