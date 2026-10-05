#!/bin/bash
# build-sweep.sh <tag> <keep> <cfg.csv>... [commit]: a SWEEP BUILD for the sampled parameter search (tools/sweep.py).
# An export of the commit (default HEAD) as in build-both.sh, then, in the export only (never the working copy):
#   1. compile check: every configuration is compiled alone-standing with the build image's glslc through the
#      unmodified gen_vulkan_spv.py (a wrapper glslc turns a failed compile into an empty .spv and a line in
#      glslc-fail.log, so one bad variant does not end the run); the exact shared-memory size of each variant is
#      read from its SPIR-V. sweep/<tag>/checked.csv = the configurations with compiled / shared_bytes / legal /
#      run (the first <keep> legal ones in file order, 0 = all);
#   2. sweep.py overlay: the variants to run, their candidate rows and SDPA profiles;
#   3. sarc/tools/build.sh (backend + tests, no llama_main) into build/<tag>; the 53 shipped SPIR-V variants must
#      still match sarc/golden/spirv.json.
# build/<tag>.src.txt records `sweep_overlay=` with the sha256 of checked.csv: a sweep build is a measurement
# build, never a candidate build (stage.sh needs llama_main, which it does not have). No GPU is used.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
TAG=${1:?tag}; KEEP=${2:?keep}; shift 2; CFGS=(); REV=HEAD
for a in "$@"; do if [[ -f $a ]]; then CFGS+=("$(readlink -f "$a")"); else REV=$a; fi; done
[[ ${#CFGS[@]} -gt 0 ]] || { echo "no configuration csv" >&2; exit 2; }
IMAGE=${SARC_BUILD_IMAGE:-localhost/et-vk-build:rocky10}
SHA=$(git -C $ET rev-parse --verify "$REV^{commit}") || { echo "unknown commit $REV" >&2; exit 2; }
P=$A/build/$TAG.src.txt; SRC=$A/src/$TAG/executorch; W=$A/sweep/$TAG
[[ -e $P || -e $SRC || -e $W ]] && { echo "tag $TAG already exists; use a new tag" >&2; exit 2; }
podman image exists $IMAGE || { echo "missing image $IMAGE" >&2; exit 2; }
mkdir -p $A/build $A/src $W
export XE2_EXPORT_MANIFEST=$A/src/$TAG.export-manifest
export_commit $ET $SHA $SRC || { echo "export failed" >&2; exit 2; }
python3 $TOOLS/sweep.py precheck-dir $SRC $W/glsl "${CFGS[@]}" || exit 2
cat > $W/glslc.sh <<'G'
#!/bin/bash
# glslc wrapper of the compile check: a failed compile leaves an empty output and a log line, status 0.
out=; prev=; for a in "$@"; do [[ $prev == -o ]] && out=$a; prev=$a; done
/usr/bin/glslc "$@" 2> "$out.err" && { rm -f "$out.err"; exit 0; }
: > "$out"; echo "$out: $(head -c 300 "$out.err" | tr '\n' ' ')" >> "$(dirname "$0")/glslc-fail.log"; exit 0
G
chmod +x $W/glslc.sh
podman run --rm --userns=keep-id --security-opt label=disable -v "$XE2_ROOT:$XE2_ROOT" -e PYTHONPATH=$(dirname $SRC) $IMAGE \
  python3 $SRC/backends/vulkan/runtime/gen_vulkan_spv.py --glsl-path $W/glsl --output-path $W/out --glslc-path=$W/glslc.sh \
  --tmp-dir-path=$W/out/shader_cache/ --optimize --nthreads 20 > $W/precheck.log 2>&1; echo "precheck generator rc=$?" >> $W/precheck.log
python3 $TOOLS/sweep.py lds $W/out $W/checked.csv $KEEP "${CFGS[@]}" | tee $W/lds.txt || exit 2
python3 $TOOLS/sweep.py overlay $SRC $W/checked.csv | tee -a $W/lds.txt || exit 2
{ echo "tag=$TAG"; echo "sweep_overlay=$W/checked.csv sha256=$(sha256sum < $W/checked.csv | cut -c1-64) (LOCAL, UNCOMMITTED variants: sweep build, not a candidate build)"
  cat $W/lds.txt; echo "commit=$SHA"; echo "tree=$(git -C $ET rev-parse $SHA^{tree})"; echo "subject=$(git -C $ET log -1 --format=%s $SHA)"
  echo "image=$IMAGE $(podman image inspect --format '{{.Id}}' $IMAGE)"; echo "start=$(date -u +%FT%TZ)"; } > $P 2>&1
ok=1
$ET/sarc/tools/build.sh $SRC $A/build/$TAG > $A/build/$TAG.log 2>&1; rc=$?
grep -q SARC_BUILD_OK $A/build/$TAG.log || rc=${rc/#0/1}; echo "main_rc=$rc" >> $P; [[ $rc == 0 ]] || ok=0
if [[ $ok == 1 ]]; then
  python3 $ET/sarc/tools/spirv_golden.py $A/build/$TAG/backend/vulkan_compute_shaders $ET/sarc/golden/spirv.json > $A/build/$TAG.golden.txt 2>&1
  rc=$?; echo "golden_rc=$rc ($(tail -1 $A/build/$TAG.golden.txt))" >> $P; [[ $rc == 0 ]] || ok=0
  sha256sum $A/build/$TAG/tests/test_llama_microbench >> $P 2>&1 || ok=0
fi
echo "end=$(date -u +%FT%TZ)" >> $P
if [[ $ok == 1 ]]; then echo BUILD_SWEEP_OK >> $P; else echo BUILD_SWEEP_FAILED >> $P; fi
tail -6 $P; [[ $ok == 1 ]]
