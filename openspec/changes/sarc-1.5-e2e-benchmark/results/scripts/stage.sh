#!/bin/bash
# stage.sh <gpu>: stock = previous best WMMA kernels (with the opt-in env they needed), sarc = re-tuned defaults.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); G=$1; RD=et-refine-2026-09-28
case $G in
  780m)   H=rocky-ryzen;     BEF=780m-before;   AFT=780m-after;   ENV="" ;;
  b580)   H=fedora;          BEF=xe2-before;    AFT=b580-after;   ENV="ET_VK_TEXTURE_COOPMAT=1" ;;
  b70)    H=fedora-gpu-eval; BEF=xe2-before;    AFT=b70-after;    ENV="ET_VK_TEXTURE_COOPMAT=1" ;;
  4070ti) H=gpu-dev-4004;    BEF=4070ti-before; AFT=4070ti-after; ENV="ET_VK_COOPMAT_ANY_DEVICE=1 ET_VK_TEXTURE_COOPMAT=1" ;;
  orin)   H=duck-naughty;    BEF=orin-before;   AFT=orin-after;   ENV="ET_VK_COOPMAT_ANY_DEVICE=1 ET_VK_TEXTURE_COOPMAT=1" ;;
esac
S=$R/stage/$G; rm -rf "$S"; mkdir -p "$S/stock" "$S/sarc"
cp $R/tools/{e2e.sh,prompt_real_2048.txt,prompt_2048.txt,prompt_check.txt,wrap.sh,optin_check.sh} "$S/"
for pair in stock:$BEF sarc:$AFT; do b=${pair%%:*}; n=${pair#*:}
  if [[ $G == orin ]]; then cp $R/build/$n/bundle/{llama_main,libllama_runner.so} "$S/$b/"
  else cp $R/build/x86-$n/llama/examples/models/llama/llama_main "$S/$b/"
       f=$(find $R/build/x86-$n/llama/examples/models/llama -name libllama_runner.so | head -1); [[ -n $f ]] && cp "$f" "$S/$b/"; fi
  cat $R/src/$n/COMMIT 2>/dev/null > "$S/$b/COMMIT" || true
done
[[ -n $ENV ]] && tr ' ' '\n' <<< "$ENV" > "$S/stock/env"
cat > "$S/STAGE.md" <<EOT
stock = PREVIOUS best WMMA kernels: $BEF @ $(cat $R/src/$BEF/COMMIT 2>/dev/null || cat $R/build/$BEF/source/../COMMIT 2>/dev/null), env: ${ENV:-none (default path)}
sarc  = RE-TUNED kernels (defaults, no env): $AFT @ $(cat $R/src/$AFT/COMMIT 2>/dev/null)
EOT
if [[ $H == fedora ]]; then mkdir -p ~/$RD; rsync -a "$S/" ~/$RD/; else rsync -a "$S/" $H:$RD/; fi
echo "staged $G -> $H:~/$RD"; cat "$S/STAGE.md"
