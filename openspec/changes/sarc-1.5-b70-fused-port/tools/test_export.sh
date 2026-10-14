#!/bin/bash
# test_export.sh: regression of host.sh export_commit (the source export build-both.sh builds from), on a
# scratch repository with a nested submodule; no GPU, no build. The live trees are made dirty in every way
# (modified and untracked files in the superproject, the submodule and the nested submodule; the submodule
# checked out at a different commit than the pinned one) and the export must still be exactly the commits.
set -u
T=$(dirname "$(readlink -f "$0")"); W=$(mktemp -d); fails=0; . $T/host.sh
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1   [$2]"; fails=$((fails + 1)); fi; }
g() { git -c user.name=t -c user.email=t@t -c protocol.file.allow=always -c init.defaultBranch=main "$@"; }
mk() { mkdir -p $W/$1 && g -C $W/$1 init -q && echo "$1 v1" > $W/$1/f.txt && g -C $W/$1 add -A && g -C $W/$1 commit -qm v1; }
mk inner; mk mid; mk top
g -C $W/mid submodule --quiet add $W/inner deep/inner && g -C $W/mid commit -qm "add inner"
g -C $W/top submodule --quiet add $W/mid third/mid && g -C $W/top commit -qm "add mid"
g -C $W/top submodule --quiet update --init --recursive
PIN=$(git -C $W/top rev-parse HEAD)
# dirt: a newer commit in the superproject, a different commit checked out in the submodule, edits and untracked files
echo "top v2" > $W/top/f.txt; g -C $W/top commit -qam v2
echo "mid v2" > $W/top/third/mid/f.txt; g -C $W/top/third/mid commit -qam v2
echo dirty >> $W/top/f.txt; echo dirty >> $W/top/third/mid/deep/inner/f.txt
echo x > $W/top/untracked.txt; echo x > $W/top/third/mid/untracked.txt; echo x > $W/top/third/mid/deep/inner/untracked.txt
export XE2_EXPORT_MANIFEST=$W/manifest
export_commit $W/top $PIN $W/out; ok "export succeeds" "[[ $? == 0 ]]"
ok "superproject file is the pinned commit's" '[[ $(cat $W/out/f.txt) == "top v1" ]]'
ok "submodule file is the pinned commit's, not the checked-out one" '[[ $(cat $W/out/third/mid/f.txt) == "mid v1" ]]'
ok "nested submodule file is the pinned commit's, not the edited one" '[[ $(cat $W/out/third/mid/deep/inner/f.txt) == "inner v1" ]]'
ok "no untracked file and no .git exported" '[[ -z $(find $W/out -name untracked.txt -o -name .git) ]]'
ok "manifest lists the three trees with their pinned commits" '[[ $(wc -l < $W/manifest) == 3 ]] && grep -q "^. $PIN$" $W/manifest && grep -q "^./third/mid/deep/inner $(git -C $W/inner rev-parse HEAD)$" $W/manifest'
ok "live trees untouched" '[[ $(tail -1 $W/top/f.txt) == dirty && -e $W/top/third/mid/untracked.txt && $(git -C $W/top status --porcelain | wc -l) -ge 2 ]]'
rm -rf $W/top/third/mid/deep/inner/.git; export_commit $W/top $PIN $W/out2 2> $W/err; ok "an uninitialised pinned submodule fails the export" "[[ $? != 0 ]] && grep -q 'not initialised' $W/err"
[[ $fails == 0 ]] && { echo TEST_EXPORT_PASS; rm -rf $W; exit 0; }
echo "TEST_EXPORT_FAIL ($fails), evidence kept in $W"; exit 1
