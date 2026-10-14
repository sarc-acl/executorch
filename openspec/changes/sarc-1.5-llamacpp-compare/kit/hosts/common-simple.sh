# Shared by the adapters whose campaign has no Intel-style host.sh: a foreign-workload check by program name and
# a guard that refuses to start beside one and records one seen at the end.
gpu_shared() { :; }
drm_clients() { :; }
gpu_others() { local p q mine
  for p in $(pgrep -x 'llama-server|ollama|llama_main|test_llama_micr|llama-completio|llama-bench|vllm|logits_dump'; pgrep -f 'ComfyUI|comfyui'); do
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$TOP" || $q == "$$" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done 2>/dev/null; }
guarded() { local f=$1 o rc; shift
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; return 76; }
  : > "$f"; "$@"; rc=$?
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; return 76; }
  return $rc; }
