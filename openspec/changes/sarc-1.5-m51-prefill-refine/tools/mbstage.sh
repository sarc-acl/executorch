#!/bin/bash
# mbstage.sh <stage name> <build tag>: a microbench-only stage, stage/m51-LOCAL-ONLY/<stage name>/ with the
# test_llama_microbench wrapper (adbshim.sh), and the build's test_llama_microbench on the board in
# $DEV_ROOT/stage/<stage name>/top. Runs: cd <stage>; env ... ./test_llama_microbench <args>.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
S=$ART/stage/$LOC/$1; B=$ART/build/m51/$2/sarc_dev/test_llama_microbench; DS=$DEV_ROOT/stage/$1/top
[[ -e $S ]] && { echo "$S exists"; exit 1; }
mkdir -p "$S"
printf '#!/bin/bash\nexec %s test_llama_microbench "$@"\n' "$TOOLS/adbshim.sh" > "$S/test_llama_microbench"; chmod +x "$S/test_llama_microbench"
A shell "mkdir -p $DS" < /dev/null; A push "$B" "$DS/" > /dev/null; A shell "chmod 755 $DS/test_llama_microbench" < /dev/null
{ echo "stage $1 from build $2 ($(sed -n 's/^tag \S* commit //p' $ART/build/m51/$2.txt)) $(date -u +%FT%TZ)"
  echo "host $(sha256sum "$B" | cut -d' ' -f1)"; echo "board $(A shell "sha256sum $DS/test_llama_microbench" < /dev/null | cut -d' ' -f1)"; } > "$S/STAGE.md"
cat "$S/STAGE.md"
