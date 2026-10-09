#!/bin/bash
# gate_sdpa.sh <session>: the gate for a candidate that changes an SDPA kernel = gate.sh --sdpa
# (12 passes each of SDPA tiers all, extended and full, 0 mismatches and pairing=ok, then the common gate).
[[ $# == 1 ]] || { echo "usage: gate_sdpa.sh <session> (the environment comes from cand/env)" >&2; exit 2; }
exec "$(dirname "$(readlink -f "$0")")/gate.sh" "$1" --sdpa
