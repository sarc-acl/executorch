#!/bin/bash
# mktree.sh <commit> <dest>/executorch: clean `git archive` of <commit> of THIS working copy plus every submodule
# (recursively) at the commit pinned by <commit>, taken from the working copy's own submodule repositories.
# Local replacement of sarc-1.5-e2e-benchmark/kit/host/mktree.sh, which reads another checkout. Nothing is
# fetched: a pinned commit that is missing locally is an error. The destination must not exist.
set -euo pipefail
source "$(dirname "$0")/common.sh"; C=$(git -C $ET rev-parse --verify "$1^{commit}"); DST=$2
[[ $(basename "$DST") == executorch ]] || { echo "destination must be named executorch" >&2; exit 2; }
[[ -e $DST ]] && { echo "$DST exists; trees are immutable, use a new tag" >&2; exit 2; }
mkdir -p "$DST"; git -C $ET archive $C | tar -x -C "$DST"
: > "$DST/../SUBMODULES"
sub() { # sub <repo> <tree-ish> <dest>: archive the submodules pinned by <tree-ish>, depth first
  local repo=$1 dst=$3 sha path
  git -C "$repo" ls-tree -r "$2" | awk '$2 == "commit" {print $3, $4}' | while read -r sha path; do
    git -C "$repo/$path" cat-file -e "$sha^{commit}" 2>/dev/null || { echo "submodule $path: pinned $sha not present locally" >&2; exit 3; }
    mkdir -p "$dst/$path"; git -C "$repo/$path" archive "$sha" | tar -x -C "$dst/$path"
    echo "$sha ${dst#"$DST"}/$path" | sed 's| /| |' >> "$DST/../SUBMODULES"
    sub "$repo/$path" "$sha" "$dst/$path"
  done
}
sub $ET $C "$DST"
echo "$C" > "$DST/../COMMIT"; chmod -R a-w "$DST"
echo "tree $C -> $DST ($(wc -l < "$DST/../SUBMODULES") submodules)"
