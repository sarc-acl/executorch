#!/bin/bash
# others.sh [pid...]: foreign GPU users and host builds, one field each, ignoring the given pids and their children:
#   gpu=<pid:comm;...>   holders of /dev/dri/{renderD128,card1} (fuser) and processes whose comm is a GPU runner
#                        (comm is the program's own name: an `adb shell ... llama_main` is "adb", not a runner)
#   build=<pid:comm;...> compiler, linker and build-driver processes of anyone on this host
ign=" $* "; for p in "$@"; do ign+=" $(pgrep -d ' ' -P $p) "; done
g=""
for p in $(fuser /dev/dri/renderD128 /dev/dri/card1 2>/dev/null) $(pgrep -x 'llama_main|test_llama_micr|llama-server|llama-bench|llama-cli|ollama|vulkaninfo'); do
  [[ $ign == *" $p "* || $g == *"$p:"* ]] && continue; g+="$p:$(cat /proc/$p/comm 2>/dev/null);"
done
b=$(pgrep -l -x 'cc1|cc1plus|ld|ld.gold|ld.lld|collect2|ninja|make|clang|clang\+\+|glslc|rustc' | head -5 | tr ' \n' ':;')
echo "gpu=$g build=$b"
