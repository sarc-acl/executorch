#!/bin/bash
# mktree_from.sh <commit> <dest>/executorch <base tag>: the same tree mktree.sh makes for <commit>, built from the
# tree of an existing build tag instead of from scratch (on this NAS a fresh archive of the 30 submodules takes
# over half an hour). Unchanged files are hard links to the base tree (both trees are read-only); every path that
# differs between the base commit and <commit> is removed and written from `git show`. Refused when a submodule
# pin differs. Checked afterwards, not assumed: the file list of the main repository equals `git ls-tree -r`
# (submodule paths excluded) and every changed file has the blob hash of <commit>.
set -euo pipefail
source "$(dirname "$0")/common.sh"; C=$(git -C $ET rev-parse --verify "$1^{commit}"); DST=$2; BT=$3
BASE=$A/build/$BT/source; BC=$(cat $BASE/COMMIT)
[[ $(basename "$DST") == executorch ]] || { echo "destination must be named executorch" >&2; exit 2; }
[[ -e $DST ]] && { echo "$DST exists; trees are immutable, use a new tag" >&2; exit 2; }
sm() { git -C $ET ls-tree -r "$1" | awk '$2 == "commit" {print $3, $4}'; }
[[ $(sm $BC) == $(sm $C) ]] || { echo "submodule pins differ between $BC and $C: use mktree.sh" >&2; exit 3; }
mkdir -p "$(dirname "$DST")"; cp -al $BASE/executorch "$DST"; cp $BASE/SUBMODULES "$DST/../SUBMODULES"
find "$DST" -type d -exec chmod u+w {} +
# The base build left python caches in its tree (directories stay writable for that): not part of the commit.
find "$DST" -depth -type d -name __pycache__ -exec rm -rf {} +
n=0
while IFS=$'\t' read -r st p p2; do
  case $st in
    D) rm -f "$DST/$p" ;;
    R*) rm -f "$DST/$p"; p=$p2 ;&
    *) rm -f "$DST/$p"; mkdir -p "$(dirname "$DST/$p")"; git -C $ET show "$C:$p" > "$DST/$p"
       [[ $(git -C $ET ls-tree "$C" -- "$p" | cut -c1-6) == 100755 ]] && chmod +x "$DST/$p"
       [[ $(git -C $ET hash-object "$DST/$p") == $(git -C $ET rev-parse "$C:$p") ]] || { echo "blob mismatch: $p" >&2; exit 4; } ;;
  esac; n=$((n + 1))
done < <(git -C $ET diff --name-status $BC $C)
find "$DST" -depth -type d -empty -delete
SUBS=$(sm $C | cut -d' ' -f2 | sed 's|^|./|; s|$|/|')
diff <(git -C $ET ls-tree -r --name-only $C | grep -vxF -f <(sm $C | cut -d' ' -f2) | sort) \
     <(cd "$DST" && find . \( -type f -o -type l \) | grep -vF -f <(echo "$SUBS") | sed 's|^\./||' | sort) > "$DST/../filelist.diff" \
  || { echo "file list differs from git ls-tree, see $DST/../filelist.diff" >&2; exit 5; }
echo "$C" > "$DST/../COMMIT"; chmod -R a-w "$DST" 2>/dev/null || true
echo "tree $C -> $DST (hard links to $BT, $n changed paths written, file list checked)"
