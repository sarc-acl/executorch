#!/bin/bash
# chain10.sh: the units of chain 8 that the reboot of 2026-10-09 01:17 UTC cut off (an orderly reboot of the
# desktop, kernel 7.2.8 -> 7.2.9; chain 8 was in the logits probe of s2-c1, chain 9 was waiting), then chain 9
# unchanged. Nothing already measured is repeated: probe.sh skips the prompts whose logits are saved (all saved
# files have the full size), the gate, the extra tiers and c1-ref6 were complete.
#   1. probe.sh s2-c1 (resumes), decide.py --arithmetic, decode_ab.sh, collect.sh; writes CHAIN8_DONE s2-c1
#   2. chain9.sh
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=s2-c1; D=$A/stage/$S; REF=c1-ref6
ST=$A/logs/chain8-$S.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain10 resumes chain8 after the reboot of 01:17 UTC (kernel $(uname -r), $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | sed 's/.*= //'))"
hold_wait "probe $S"
$TOOLS/probe.sh $S > $A/logs/probe-$S.resume.out 2>&1; say "probe $S rc=$? $(tail -1 $D/probe/analysis.txt 2>/dev/null)"
python3 $TOOLS/decide.py $D --arithmetic $A/raw/$REF/full.csv > $A/logs/decide-$S.out 2>&1; say "decide $S rc=$? $(head -1 $D/decision.txt 2>/dev/null)"
hold_wait "decode $S"
$TOOLS/decode_ab.sh $S > $A/logs/decode-$S.out 2>&1; say "decode $S rc=$?"
$TOOLS/collect.sh > $A/logs/collect-$S.out 2>&1; say "collect rc=$?"
say "CHAIN8_DONE $S"
hold_wait "chain9"
exec $TOOLS/chain9.sh
