#!/bin/bash
# Host wrapper: stop the GPU's co-tenant services for the whole campaign, run e2e.sh,
# then the dispatch trace (trace.sh), and restore the services on exit.
# usage: wrap.sh <e2e.sh args...>   (same args; --gpu selects the services)
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
GPU=$(sed -n 's/.*--gpu \([^ ]*\).*/\1/p' <<< "$*")
: > services-stopped.txt
case $GPU in
  4070ti) UNITS=$(systemctl list-units --type=service --state=running --no-pager --plain | awk '{print $1}' | grep -E '^zun-flux-pipeline\.service$') ;;
  b70)    UNITS=$(systemctl list-units --type=service --state=running --no-pager --plain | awk '{print $1}' | grep -E 'llm-api') ;;
  *)      UNITS="" ;;
esac
for s in $UNITS; do sudo -n systemctl stop "$s" && echo "$s" >> services-stopped.txt; done
restore() { for s in $(cat services-stopped.txt); do sudo -n systemctl start "$s"; done
  echo "restored: $(tr '\n' ' ' < services-stopped.txt)"; }
trap restore EXIT
echo "stopped: $(tr '\n' ' ' < services-stopped.txt)"; sleep 10
./e2e.sh "$@" --prompt prompt_real_2048.txt --out raw_real --no-check
./probe.sh "$@"
echo WRAP3_DONE
