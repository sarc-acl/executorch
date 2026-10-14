#!/bin/bash
# spv_identity.sh <topic build tag> <parent build tag> <B580 build dir>: byte comparison of compiled SPIR-V.
#   1. every shader of build/<parent> exists in build/<topic> with the same bytes (nothing that existed changed);
#   2. every sarc_dev_b580_sdpa_{fused,kvt}* shader of build/<topic> has the same bytes as in the B580 build.
# Reads backend/vulkan_compute_shaders/*.spv of each build. Prints counts and SPV_IDENTITY_OK, or the names that
# differ and SPV_IDENTITY_DIFFERENT; exit status follows. No GPU.
. "$(dirname "$(readlink -f "$0")")/host.sh"; T=$A/build/${1:?topic}/backend/vulkan_compute_shaders; P=$A/build/${2:?parent}/backend/vulkan_compute_shaders; B=${3:?B580 build}/backend/vulkan_compute_shaders
bad=0; np=0; nb=0
for f in $P/*.spv; do n=${f##*/}; np=$((np + 1)); cmp -s $f $T/$n || { echo "parent shader differs or is missing in the topic build: $n"; bad=$((bad + 1)); }; done
for f in $T/sarc_dev_b580_sdpa_fused*.spv $T/sarc_dev_b580_sdpa_kvt*.spv; do n=${f##*/}; nb=$((nb + 1)); cmp -s $f $B/$n || { echo "differs from the B580 build or is missing there: $n"; bad=$((bad + 1)); }; done
echo "parent shaders compared: $np; topic shaders in all: $(ls $T/*.spv | wc -l); fused and copy-pass shaders compared with $3: $nb"
echo "sha256 of the sorted list of (sha256, name) of the fused and copy-pass shaders: $(cd $T && sha256sum sarc_dev_b580_sdpa_fused*.spv sarc_dev_b580_sdpa_kvt*.spv | sha256sum | cut -c1-64)"
[[ $bad == 0 && $np -gt 0 && $nb -gt 0 ]] && { echo SPV_IDENTITY_OK; exit 0; }; echo "SPV_IDENTITY_DIFFERENT ($bad)"; exit 1
