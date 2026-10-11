#!/bin/bash
# f1-decode.sh [processes per arm = 10]: device side. POST HOC, not pre-registered: written after the kernel table
# of f1-chain.sh step 5 showed the decode-size (M = 1) stock SDPA cases of one model per environment outside the
# band, with per-process values on separate levels inside both arms. More processes of the same suite restricted
# to those cases (--sdpa --regime=decode), old / new interleaved (old new new old ...), both environments, with
# the devfreq clock read before and after each process. It replaces nothing: the table of step 5 stands as measured.
cd "$(dirname "$0")"; source ./common.sh
N=${1:-10}; OLD=topic4; NEW=f1; C1="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1"; O=$A/raw/kernel-decode; mkdir -p $O
for e in fused1 noenv; do E=""; [[ $e == fused1 ]] && E=$C1
  for r in $(seq 1 $N); do
    if (( r % 2 )); then order="$OLD $NEW"; else order="$NEW $OLD"; fi
    for b in $order; do cool_start 120; c0=$(gclk)
      env $E LD_LIBRARY_PATH=$A/build/$b/bundle ./gl.sh $A/build/$b/bundle/test_llama_microbench --sdpa --regime=decode --json-out=$O/$e-$b-sdpa-r$r.json > $O/$e-$b-sdpa-r$r.log 2>&1
      rc=$?; echo "$e,$b,sdpa-decode,$r,$rc,$(date -u +%FT%TZ),$c0,$(gclk)" >> $O/rc.csv; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "STOPPED rc=$rc"; exit $rc; }
    done
  done
done
echo "DECODE_DONE $(wc -l < $O/rc.csv) runs"
