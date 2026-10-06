#!/bin/bash
# sweep_incumbent.sh <space> <fam:M:N:K:sx:sy:S>... [ref=<label>:<token> ...]: the incumbents as members of the
# sampled search (a queue job, after sweep_stage2.sh). The refinement of the owner's procedure starts from the
# best 20 of the uniform sample; where the kernels already in use are faster than everything in the sample
# (8da4w: the candidate 2 tile is 1.20x of the shipped one, the best of the sample 0.97x), that refinement never
# visits their neighbourhood. This job adds the named incumbent configurations themselves and every legal
# one-parameter neighbour of them that is not measured yet as the sweep build sw2i-<space> (ids from 50000) and
# screens them in the cheap mode. The later steps (sweep_rounds.sh: correctness, refinement rounds by the 2 %
# rule, full x 2 confirmation) then treat them like any other configuration. An addition to the procedure, not
# a replacement: the sample, its analysis and its refinement are unchanged.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; shift; C=$A/sweep/cfg; INC=(); REFS=()
for a in "$@"; do if [[ $a == ref=* ]]; then REFS+=("$a"); else INC+=("$a"); fi; done
if [[ ! -f $A/build/sw2i-$S.src.txt ]]; then
  { head -1 $A/sweep/sw1-$S/checked.csv; for b in sw1 sw2; do tail -n +2 $A/sweep/$b-$S/checked.csv; done; } > $C/$S-measuredi.csv
  python3 - $TOOLS $S $C/$S-measuredi.csv $C/$S-incumbents.csv "${INC[@]}" <<'P' || exit 1
import sys; sys.path.insert(0, sys.argv[1]); import sweep
space, measured, out, inc = sys.argv[2], sys.argv[3], sys.argv[4], [tuple(a.split(":")) for a in sys.argv[5:]]
L = list(sweep.LEGAL[space]()); geo = lambda c: tuple(str(c[k]) for k in ("fam", "M", "N", "K", "sx", "sy", "S"))
seeds = [c for c in L if geo(c) in inc]; assert len(seeds) == len(inc), (inc, [geo(c) for c in seeds])
have = {sweep.key(c, space) for c in sweep.read([measured])}; bk = [sweep.key(c, space) for c in seeds]
rows = [c for c in seeds if sweep.key(c, space) not in have]
rows += [c for c in L if sweep.key(c, space) not in have and any(sum(a != b for a, b in zip(sweep.key(c, space), k)) == 1 for k in bk)]
for i, c in enumerate(rows): c["id"] = f"{sweep.SPACES[space]}{50000 + i:05d}"
sweep.write(out, rows); print(f"{space}: {len(seeds)} incumbents, {len(rows)} configurations to add -> {out}")
P
  bash $TOOLS/build-sweep.sh sw2i-$S 0 $C/$S-incumbents.csv || exit 1
fi
bash $TOOLS/sweep_screen.sh sw2i-$S $S $A/sweep/sw2i-$S/checked.csv "${REFS[@]}"
