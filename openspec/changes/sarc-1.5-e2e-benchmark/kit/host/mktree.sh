#!/bin/bash
# mktree.sh <commit> <dest>/executorch : clean git archive of <commit> plus each submodule at the
# commit pinned by <commit>, taken from the main clone's submodule repos (fetches if missing).
set -euo pipefail
REPO=~/Desktop/sarc-acl/main/executorch; C=$1; DST=$2
mkdir -p "$DST"; git -C $REPO archive $C | tar -x -C "$DST"
git -C $REPO ls-tree -r $C | awk '$2=="commit"{print $3, $4}' | while read -r sha path; do
  sub=$REPO/$path
  if ! git -C "$sub" cat-file -e "$sha^{commit}" 2>/dev/null; then git -C "$sub" fetch -q origin "$sha" 2>/dev/null || git -C "$sub" fetch -q origin; fi
  mkdir -p "$DST/$path"; git -C "$sub" archive "$sha" | tar -x -C "$DST/$path"
  # nested submodules
  git -C "$sub" ls-tree -r "$sha" | awk '$2=="commit"{print $3, $4}' | while read -r s2 p2; do
    n=$sub/$p2; [[ -d $n ]] || { echo "WARN nested $path/$p2 missing"; continue; }
    git -C "$n" cat-file -e "$s2^{commit}" 2>/dev/null || git -C "$n" fetch -q origin 2>/dev/null || true
    mkdir -p "$DST/$path/$p2"; git -C "$n" archive "$s2" | tar -x -C "$DST/$path/$p2" || echo "WARN nested archive $path/$p2"
  done
done
echo "$C" > "$(dirname "$DST")/COMMIT"; echo "tree $C -> $DST"
