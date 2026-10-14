#!/bin/bash
# export_commit.sh <commit> <dest>: writes <dest>/executorch = exactly <commit> of this repository and, recursively,
# every submodule at the commit its parent pins, read from the git object stores (never from the working tree).
# Writes <dest>/MANIFEST.txt: one line "<path> <commit>" per exported repository. Fails if an object is missing.
set -euo pipefail
REPO=$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel); C=$(git -C "$REPO" rev-parse "$1^{commit}"); D=$2
[[ -e $D ]] && { echo "$D exists: a build tag is exported once" >&2; exit 1; }
mkdir -p "$D/executorch"; D=$(cd "$D" && pwd)
exp() { # exp <git dir> <commit> <relative path>
  local g=$1 c=$2 p=$3
  git --git-dir="$g" archive --format=tar "$c" | tar -x -C "$D/executorch/$p"
  echo "${p:-.} $c" >> "$D/MANIFEST.txt"
  git --git-dir="$g" ls-tree -r "$c" | awk '$2 == "commit" {print $3, $4}' | while read -r sc sp; do
    local name sg
    name=$(git --git-dir="$g" config --blob "$c:.gitmodules" --get-regexp '^submodule\..*\.path$' | awk -v p="$sp" '$2 == p {sub(/^submodule\./, "", $1); sub(/\.path$/, "", $1); print $1}')
    sg=$g/modules/$name
    [[ -d $sg ]] || { echo "missing object store for submodule $p/$sp ($name)" >&2; exit 1; }
    mkdir -p "$D/executorch/$p${p:+/}$sp"
    exp "$sg" "$sc" "$p${p:+/}$sp"
  done
}
: > "$D/MANIFEST.txt"
exp "$(git -C "$REPO" rev-parse --absolute-git-dir)" "$C" ""
echo "exported $C to $D/executorch ($(wc -l < "$D/MANIFEST.txt") repositories)"
