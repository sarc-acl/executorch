#!/bin/bash
# Campaign builds. x86 chain and Orin cross chain run concurrently.
C=/home/doremy/Desktop/sarc-acl/.artifacts/e2e-1.5-2026-09-28; B=/home/doremy/Desktop/sarc-acl/dev/1.5/executorch/sarc/tools/build.sh
x86() {
  SARC_JOBS=10 $B --llama --no-tests $C/src/stock/executorch $C/build/x86-stock > $C/build/x86-stock.log 2>&1; echo "x86-stock rc=$?"
  SARC_JOBS=10 $B --llama --no-tests $C/src/sarc/executorch $C/build/x86-sarc > $C/build/x86-sarc.log 2>&1; echo "x86-sarc rc=$?"
  SARC_JOBS=10 $B ~/Desktop/sarc-acl/dev/1.5/executorch $C/build/x86-dev > $C/build/x86-dev.log 2>&1; echo "x86-dev(tests) rc=$?"
  SARC_JOBS=10 $B --llama --no-tests --traced $C/src/sarc/executorch $C/build/x86-sarc-traced > $C/build/x86-sarc-traced.log 2>&1; echo "x86-sarc-traced rc=$?"
}
orin() {
  for n in stock sarc dev; do
    JETSON_CROSS_WORK=$C/build/orin-$n $C/tools/jetson-cross/container.sh > $C/build/orin-$n.log 2>&1; echo "orin-$n rc=$?"
  done
}
x86 & orin & wait; echo BUILD_ALL_DONE
