#!/bin/bash
# export_recursive.sh <commit> <dest dir> <store dir>: export of one commit of the working copy's repository and,
# recursively, of every submodule at the commit it pins, from git object stores only (R5; owner decision
# 2026-10-08 22:55 UTC). The repository itself is exported with `git archive` from the working copy's object
# store; each submodule (and each nested one) is fetched at exactly its pinned commit from the URL of the
# .gitmodules of the commit that pins it into a bare repository <store dir>/<path with / as __>.git, and its tree
# is written from that store (read-tree + checkout-index, so that no export-ignore attribute drops a file).
# Nothing is read from a working tree and the working copy is not touched.
# Writes <dest dir>/executorch, <dest dir>/MANIFEST (path, pinned commit, url, tree id, files, content hash) and
# <dest dir>/COMMIT. The content hash is the one chain25.sh used: sha256 over the sorted `sha256sum` of all files.
set -euo pipefail
ET=/home/doremy/hmz-sarc/executorch; C=$(git -C $ET rev-parse "$1^{commit}"); D=$2; ST=$3
[[ -e $D ]] && { echo "REFUSED: $D exists"; exit 2; }
mkdir -p $D/executorch $ST
git -C $ET archive $C | tar -x -C $D/executorch
echo ". $C source=git-archive store=$ET/.git tree=$(git -C $ET rev-parse $C^{tree})" > $D/MANIFEST
sub() {  # sub <git dir of the parent> <parent commit> <path prefix in the export>
  local G=$1 PC=$2 PRE=$3 sha path url name B
  git --git-dir=$G ls-tree -r $PC | awk '$2=="commit"{print $3, $4}' | while read -r sha path; do
    name=$(git --git-dir=$G config --blob $PC:.gitmodules --get-regexp '^submodule\..*\.path$' | awk -v p="$path" '$2==p{print $1}' | sed 's/^submodule\.//; s/\.path$//')
    url=$(git --git-dir=$G config --blob $PC:.gitmodules --get "submodule.$name.url")
    B=$ST/$(echo "$PRE$path" | sed 's|/|__|g').git
    [[ -d $B ]] || git init -q --bare $B
    git --git-dir=$B cat-file -e "$sha^{commit}" 2>/dev/null || git --git-dir=$B fetch -q --depth 1 "$url" "$sha"
    [[ $(git --git-dir=$B rev-parse "$sha^{commit}") == "$sha" ]]
    git --git-dir=$B fsck --no-dangling --no-progress "$sha" > /dev/null
    mkdir -p "$D/executorch/$PRE$path"
    [[ -z $(ls -A "$D/executorch/$PRE$path") ]] || { echo "not empty: $PRE$path"; exit 3; }
    GIT_INDEX_FILE=$B/export.index git --git-dir=$B --work-tree="$D/executorch/$PRE$path" read-tree "$sha"
    GIT_INDEX_FILE=$B/export.index git --git-dir=$B --work-tree="$D/executorch/$PRE$path" checkout-index -a -f
    rm -f $B/export.index
    echo "$PRE$path pinned=$sha source=object-store store=$B url=$url tree=$(git --git-dir=$B rev-parse $sha^{tree}) files=$(cd "$D/executorch/$PRE$path" && find . -type f -o -type l | wc -l) files_sha256=$(cd "$D/executorch/$PRE$path" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)" >> $D/MANIFEST
    sub $B $sha "$PRE$path/"
  done
}
sub $ET/.git $C ""
echo $C > $D/COMMIT; echo "EXPORT_DONE $(wc -l < $D/MANIFEST) manifest entries"
