#!/usr/bin/env bash
# extra.sh (inside the cross container, /work = an existing build tag): logits_dump for the logits probe, built
# against that tag's install prefix, into /work/bundle. The main build is not touched.
set -euo pipefail
common=(-DCMAKE_TOOLCHAIN_FILE=/recipe/aarch64.cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/venv/bin/python)
cmake -S /recipe/logits_dump -B /work/logits_dump -G Ninja "${common[@]}" -DEXECUTORCH_ROOT=/work/source/executorch \
  -DCMAKE_PREFIX_PATH=/work/build -DCMAKE_FIND_ROOT_PATH='/work/build;/usr/aarch64-linux-gnu'
cmake --build /work/logits_dump -j8
cp /work/logits_dump/logits_dump /work/bundle/; file /work/bundle/logits_dump
