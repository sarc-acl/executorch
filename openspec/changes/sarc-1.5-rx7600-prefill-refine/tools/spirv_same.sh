#!/bin/bash
# spirv_same.sh <build tag>: the shipped-SPIR-V checks of a native build.
#  1. sarc/tools/spirv_golden.py against sarc/golden/spirv.json: reported, PENDING (the golden needs the
#     container's glslc; the native SDK glslc gives 14 of 53 shipped variants other bytes on the parent itself);
#  2. the same tool against $A/golden-ref-parent.json, which it wrote from the native parent build
#     (`--update --owner native-parent`, once): every shipped variant byte-identical to the parent of the same
#     toolchain. This is the check a candidate must pass.
source "$(dirname "$(readlink -f "$0")")/env.sh"
B=$A/build/rx7600/$1/llama/vulkan_compute_shaders; G=$ET/sarc/tools/spirv_golden.py
[[ -f $A/golden-ref-parent.json ]] || /usr/bin/python3 $G $A/build/rx7600/parent/llama/vulkan_compute_shaders $A/golden-ref-parent.json --update --owner native-parent > /dev/null
/usr/bin/python3 $G $B $ET/sarc/golden/spirv.json > $A/logs/golden-$1.txt 2>&1
echo "golden sarc/golden/spirv.json (PENDING, native glslc): $(tail -1 $A/logs/golden-$1.txt), $(grep -c '^DIFF' $A/logs/golden-$1.txt) DIFF"
/usr/bin/python3 $G $B $A/golden-ref-parent.json > $A/logs/golden-ref-$1.txt 2>&1
echo "shipped SPIR-V vs native parent: $(tail -1 $A/logs/golden-ref-$1.txt)"; grep -v '^spirv golden' $A/logs/golden-ref-$1.txt | head -5
