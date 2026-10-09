#!/bin/bash
# collect_session.sh <stage session> <build tag of the candidate> [dest name]: copies the small evidence files of one staged session into
# results/rx7600/sessions/<dest> (default: the session name) with the placeholders of results/rx7600/README.md (coordinator order M2c:
# no host name, no home or artifact paths in committed results): STAGE.md, gate.status, prestart.txt, raw/{runs,env,summary,nexttoken},
# verify.out / verify-compare.txt / verify.meta, trace.out and trace/{families,totals,kernels}.csv, linear-bitwise/{bitwise,kernels}.txt,
# the golden checks of the candidate build. Prints the lines that still contain a host or user name.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; TAG=$2; D=$T/../results/rx7600/sessions/${3:-$1}; mkdir -p $D
for f in STAGE.md gate.status prestart.txt verify.out verify-compare.txt verify.meta trace.out linear-bitwise.out; do [[ -f $S/$f ]] && cp -f $S/$f $D/; done
for f in runs.csv env.txt nexttoken.csv; do [[ -f $S/raw/$f ]] && cp -f $S/raw/$f $D/; done
[[ -f $S/raw/summary.csv ]] && cp -f $S/raw/summary.csv $D/
for f in families totals kernels; do [[ -f $S/trace/$f.csv ]] && cp -f $S/trace/$f.csv $D/trace-$f.csv; done
[[ -d $S/linear-bitwise ]] && { mkdir -p $D/linear-bitwise; cp -f $S/linear-bitwise/bitwise.txt $S/linear-bitwise/kernels.txt $D/linear-bitwise/ 2>/dev/null; }
{ echo "golden checks of build $TAG ($(cat $A/src/rx7600/$TAG/COMMIT))"; cat $A/logs/golden-$TAG.out; echo "-- against golden-ref-parent.json"; cat $A/logs/golden-ref-$TAG.txt; echo "-- against sarc/golden/spirv.json (PENDING for native builds)"; cat $A/logs/golden-$TAG.txt; } > $D/golden.txt 2>/dev/null
[[ -d $S/verify ]] && { mkdir -p $D/verify; cp -f $S/verify/env.txt $S/verify/correctness.log $D/verify/ 2>/dev/null; }
find $D -type f \( -name '*.txt' -o -name '*.csv' -o -name '*.md' -o -name '*.out' -o -name '*.status' -o -name '*.meta' -o -name '*.log' \) -print0 | xargs -0 sed -i \
  -e "s#$A#<artifacts>#g" -e "s#<scratch>#<scratch>#g" -e "s#<campaign-root>#<campaign-root>#g" \
  -e "s#<mesa-install>#<mesa>#g" -e "s#<workspace>/.artifacts/2026-09-28/e2eb/rx7600/models-src#<models-src>#g" \
  -e "s#<vulkan-sdk>#<vulkan-sdk>#g" -e "s#<home>#<home>#g" -e "s#host-ws1#<host>#g"
grep -rn 'yanwen\|sj1-' $D | head -5
echo "collected $(find $D -type f | wc -l) files into $D"
