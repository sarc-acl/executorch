#!/bin/bash
# mk-tools.sh: workstation side. The fused-port campaign's tools (openspec/changes/sarc-1.5-orin-fused-port/tools
# of this working copy, byte-identical to the campaign's) resolve their roots from two constants in common.sh: the
# campaign's own working copy and its directory on the device. This task may not write there, so the tools run
# from a copy (<artifacts>/tools) in which exactly those two constants point at this task's roots
# (/mnt/linux-share/hmz-campaigns/f1-orin, ~/hmz-f1-orin). Nothing else differs: data/tools.diff is the whole
# difference. f1-chain.sh, f1-decode.sh and logits_cmp.py (this task's device side) are added to the copy so that deploy.sh carries it.
set -euo pipefail
ME=$(dirname "$(readlink -f "$0")"); ET=$(cd $ME/../../../.. && pwd); CT=$ET/openspec/changes/sarc-1.5-orin-fused-port/tools
CAMPAIGN=/mnt/linux-share/hmz-campaigns/jetson-fused/executorch/openspec/changes/sarc-1.5-orin-fused-port/tools
A=/mnt/linux-share/hmz-campaigns/f1-orin/.artifacts; T=$A/tools
diff -rq -x __pycache__ $CT $CAMPAIGN || { echo "the tools of this working copy differ from the campaign's" >&2; exit 3; }
rm -rf $T; mkdir -p $T; rsync -a --exclude __pycache__ $CT/ $T/
sed -i 's|^DEVICE=doremy@duck-naughty; DEVROOT=hmz-sarc-orin-fused |DEVICE=doremy@duck-naughty; DEVROOT=hmz-f1-orin |; s|R=/mnt/linux-share/hmz-campaigns/jetson-fused;|R=/mnt/linux-share/hmz-campaigns/f1-orin;|' $T/common.sh
diff -ru $CT $T > $ME/../data/tools.diff || true
[[ $(grep -c '^[-+][^-+]' $ME/../data/tools.diff) == 4 ]] || { echo "unexpected difference, see data/tools.diff" >&2; exit 3; }
install -m 755 $ME/f1-chain.sh $ME/f1-decode.sh $ME/logits_cmp.py $T/
echo "tools copy $T: $(grep -c '^[-+][^-+]' $ME/../data/tools.diff) changed lines in common.sh, f1-chain.sh added"
