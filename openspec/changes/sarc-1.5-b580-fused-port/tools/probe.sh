#!/bin/bash
# probe.sh <session> [models=1b,3b,8b] [schemes=4w,8da4w]: logits evidence for a staged session (owner decisions
# of 2026-10-04 in CAMPAIGN.md). For every cell it runs tools/logits_probe (a fresh process per prompt, as the
# gate does) and keeps the whole last-position logits vector, for the arms
#   P  parent build, parent env            PT  the same with ET_VK_FORCE_TILED_LINEAR=1
#   C  candidate build, candidate env      CT  the same with ET_VK_FORCE_TILED_LINEAR=1 (gate prompts only)
# on the two gate prompts (the 2048-token timed prompt = 2048 x token 279 " the", and the 1972-token real-text
# check prompt) and on 35 windows of real text (two documents, lengths 256 to 2048 in steps of 256 so that the
# tile-aligned kernels run, 33 of them with a known next token). verify.sh and the prompts are not touched;
# the token ids are the kit's (sarc-1.5-e2e-benchmark/kit/logits_probe).
# Output: stage/<session>/probe/<model>-<scheme>/<arm>/<prompt>.{json,f32,log}; resumable. Then probe_analysis.py.
# The probe binaries come from build/<tag>/probe (build-probe.sh) of the two builds named in STAGE.md.
. "$(dirname "$(readlink -f "$0")")/host.sh"; SN=${1:?session}; S=$A/stage/$SN; MODELS=${2:-1b,3b,8b}; SCHEMES=${3:-4w,8da4w}
O=$S/probe; mkdir -p $O || exit 2; KIT=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/logits_probe; MROOT=/mnt/linux-share/models
PT=$(sed -n 's/^parent = build\/\([^ ]*\) .*/\1/p' $S/STAGE.md); CT=$(sed -n 's/^cand   = build\/\([^ ]*\) .*/\1/p' $S/STAGE.md)
for t in $PT $CT; do [[ -x $A/build/$t/probe/logits_probe ]] || { echo "no probe for build $t: run build-probe.sh $t" >&2; exit 2; }; done
python3 -c "print(' '.join(['279'] * 2048))" > $O/the2048_ids.txt; cp -f $KIT/real_ids.txt $KIT/check_ids.txt $O/
# prompt list: name idsfile offset length
{ echo "gate-the2048 the2048_ids.txt 0 2048"; echo "gate-check1972 check_ids.txt 0 1972"
  python3 - <<'PY'
real = {2048: [0], 1792: [0, 256], 1536: [0, 256, 512], 1280: [0, 384, 768], 1024: [0, 512, 1024], 768: [0, 640, 1280], 512: [0, 768, 1536], 256: [0, 896, 1792]}
check = {1792: [0, 128], 1536: [0, 384], 1280: [0, 640], 1024: [0, 896], 768: [0, 1152], 512: [0, 1408], 256: [0, 1664]}
for doc, f, n, w in (("gpl", "real_ids.txt", 2048, real), ("license", "check_ids.txt", 1972, check)):
    for L, offs in w.items():
        for o in offs:
            assert o + L <= n
            print(f"w{L}-{doc}-{o} {f} {o} {L}")
PY
} > $O/prompts.txt
{ date -u; hostname; echo "session $SN parent build $PT cand build $CT"; sha256sum $A/build/$PT/probe/logits_probe $A/build/$CT/probe/logits_probe $O/*_ids.txt; echo "parent env: $(tr '\n' ' ' < $S/parent/env)"; echo "cand env: $(tr '\n' ' ' < $S/cand/env)"; wc -l < $O/prompts.txt; } > $O/env.txt
cat > $O/batch.sh <<'B'
#!/bin/bash
# batch.sh <probe binary> <pte> <out dir> <ids dir> <prompt list> [gate]: one fresh process per prompt
P=$1; M=$2; D=$3; I=$4; L=$5; ONLY=${6:-}; mkdir -p $D
while read -r name f off len; do
  [[ -n $ONLY && $name != gate-* ]] && continue
  [[ -s $D/$name.f32 ]] && continue
  timeout 1800 $P $M $I/$f $off $len $D/$name > $D/$name.log 2>&1 < /dev/null; echo "RC=$?" >> $D/$name.log
done < $L
B
chmod +x $O/batch.sh
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do
  PTE=$MROOT/$MD/exported/${ST}_vulkan_$q.pte
  for arm in ${B580_PROBE_ARMS:-P C PT CT}; do   # B580_PROBE_ARMS="P C": enough to show bit-identity; the analysis then needs --pc
    case $arm in P|PT) t=$PT; e=$S/parent/env ;; *) t=$CT; e=$S/cand/env ;; esac
    benv=(); mapfile -t benv < $e; [[ $arm == *T ]] && benv+=(ET_VK_FORCE_TILED_LINEAR=1)
    gate=""; [[ $arm == CT ]] && gate=gate
    cool_start
    env "${benv[@]}" $TOOLS/gl.sh $O/batch.sh $A/build/$t/probe/logits_probe $PTE $O/$m-$q/$arm $O $O/prompts.txt $gate; rc=$?
    echo "probe $m $q $arm rc=$rc files=$(ls $O/$m-$q/$arm/*.f32 2>/dev/null | wc -l) bad=$(grep -L 'RC=0' $O/$m-$q/$arm/*.log 2>/dev/null | wc -l)"
    [[ $rc == 75 || $rc == 76 ]] && { echo "PROBE_ABORTED rc=$rc"; exit $rc; }
  done
done; done
$B580_PYTHON $TOOLS/probe_analysis.py $O $([[ ${B580_PROBE_ARMS:-} == "P C" ]] && echo --pc) > $O/analysis.txt 2>&1; echo "analysis rc=$?"; tail -30 $O/analysis.txt; echo PROBE_DONE
