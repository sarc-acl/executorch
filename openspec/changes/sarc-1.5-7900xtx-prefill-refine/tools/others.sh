#!/bin/bash
# others.sh [pid...]: foreign GPU users and host builds, one field each, ignoring the given pids and their children:
#   gpu=<pid:comm;...>   holders of /dev/dri/{renderD128,card1} that fuser can see (own processes only: the display server and
#                        other users' processes are invisible without root) and processes of anyone whose comm is a GPU runner
#                        (comm is the program's own name: an `adb shell ... llama_main` is "adb", not a runner). The idle
#                        `ollama serve` of the GPU host is NOT a runner (never stopped; GPU busy 0 %); an `ollama runner` is.
#                        Foreign use that this cannot see is bounded by the gpu_busy_percent sample before every run (e2e5.sh).
#   build=<pid:comm;...> compiler, linker and build-driver (make, gmake, cmake, ninja, ccache) processes of anyone
ign=" $* "; for p in "$@"; do ign+=" $(pgrep -d ' ' -P $p) "; done
g=""
for p in $(fuser /dev/dri/renderD128 /dev/dri/card1 2>/dev/null) $(pgrep -x 'llama_main|test_llama_micr|llama-server|llama-bench|llama-cli|vulkaninfo|igpu-roofline') $(pgrep -f 'ollama runner'); do
  [[ $ign == *" $p "* || $g == *"$p:"* ]] && continue; g+="$p:$(cat /proc/$p/comm 2>/dev/null);"
done
b=$(pgrep -l -x 'cc1|cc1plus|as|ld|ld.bfd|ld.gold|ld.lld|collect2|ninja|make|gmake|cmake|ccache|clang|clang\+\+|gcc|g\+\+|c\+\+|glslc|rustc' | head -5 | tr ' \n' ':;')
echo "gpu=$g build=$b"
