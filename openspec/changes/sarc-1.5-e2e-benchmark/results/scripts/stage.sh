#!/bin/bash
# stage.sh <gpu>: copy binaries + scripts to the GPU host's campaign dir.
set -euo pipefail
C=$(cd "$(dirname "$0")/.." && pwd); G=$1; RD=et-e2e-1.5-2026-09-28
case $G in
  780m) H=rocky-ryzen ;; 4070ti) H=gpu-dev-4004 ;; b70) H=fedora-gpu-eval ;; b580) H=fedora ;; orin) H=duck-naughty ;;
  *) echo "unknown gpu" >&2; exit 2 ;;
esac
S=$C/stage/$G; rm -rf "$S"; mkdir -p "$S"
cp $C/tools/{e2e.sh,trace.sh,wrap.sh,prompt_2048.txt,prompt_check.txt} "$S/"
for b in stock sarc; do
  mkdir -p "$S/$b"
  if [[ $G == orin ]]; then cp $C/build/orin-$b/bundle/{llama_main,libllama_runner.so} "$S/$b/"
  else
    cp $C/build/x86-$b/llama/examples/models/llama/llama_main "$S/$b/"
    cp $C/build/x86-$b/llama/examples/models/llama/runner/libllama_runner.so "$S/$b/"
    if [[ -f $C/build/x86-$b-traced/llama/examples/models/llama/llama_main && ${NO_TRACED:-0} == 0 ]]; then mkdir -p "$S/$b-traced"
      cp $C/build/x86-$b-traced/llama/examples/models/llama/llama_main "$S/$b-traced/"
      cp $C/build/x86-$b-traced/llama/examples/models/llama/runner/libllama_runner.so "$S/$b-traced/"; fi
  fi
done
if [[ $H == fedora ]]; then mkdir -p ~/$RD; rsync -a "$S/" ~/$RD/
else rsync -a "$S/" $H:$RD/; fi
echo "staged $G -> $H:~/$RD"
