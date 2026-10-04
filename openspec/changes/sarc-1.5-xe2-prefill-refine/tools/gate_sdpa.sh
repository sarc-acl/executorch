#!/bin/bash
# gate_sdpa.sh <session> "<cand env>": the gate for a candidate that changes an SDPA kernel = gate.sh --sdpa
# (12 passes each of SDPA tiers all, extended and full, 0 mismatches and pairing=ok, then the common gate).
exec "$(dirname "$(readlink -f "$0")")/gate.sh" "$1" "$2" --sdpa
