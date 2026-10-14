#!/bin/bash
# chain1.sh: device side. The two controls of this campaign on the parent build (8973ced76):
#   s0-parent-verify  unmodified verify.sh with the parent environment (the first campaign's final stack): the
#                     snapshot every candidate's verify.out is compared with, plus one SDPA pass per tier (recorded);
#   s0n-noenv         the same with nothing selected: the snapshot for the hook's "nothing selected, nothing
#                     changed" control (owner decision D4) and the pristine arm.
cd "$(dirname "$0")"; source ./common.sh
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
PENV="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"
step ./parent_verify.sh parent "$PENV"
step env PARENT_CTL_NAME=s0n-noenv ./parent_verify.sh parent ""
echo CHAIN_DONE
