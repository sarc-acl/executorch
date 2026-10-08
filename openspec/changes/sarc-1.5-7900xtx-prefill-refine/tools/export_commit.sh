#!/bin/bash
# export_commit.sh <commit> <tag>: $A/src/7900xtx/<tag>/executorch = git archive of <commit> from this clone plus,
# recursively, every submodule at the commit <commit> pins, from the submodule object stores of this clone
# (.git/modules), never from a working tree. Writes $A/src/7900xtx/<tag>/{COMMIT,MANIFEST}. A tag is exported once.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
C=$(git -C $ET rev-parse "$1^{commit}"); TAG=$2; D=$A/src/7900xtx/$TAG
[[ -e $D ]] && { echo "export $TAG exists" >&2; exit 2; }
mkdir -p $D/executorch
git -C $ET archive $C | tar -x -C $D/executorch
echo ". $C" > $D/MANIFEST
sub() { # sub <repo dir> <commit> <prefix>
  git -C "$1" ls-tree -r "$2" | awk '$2=="commit"{print $3, $4}' | while read -r sha path; do
    local r=$1/$path
    git -C "$r" cat-file -e "$sha^{commit}" || { echo "missing $3$path@$sha" >&2; exit 3; }
    mkdir -p "$D/executorch/$3$path"; git -C "$r" archive "$sha" | tar -x -C "$D/executorch/$3$path"
    echo "$3$path $sha" >> $D/MANIFEST
    sub "$r" "$sha" "$3$path/"
  done
}
sub $ET $C ""
echo $C > $D/COMMIT; echo "exported $C -> $D ($(wc -l < $D/MANIFEST) entries)"
