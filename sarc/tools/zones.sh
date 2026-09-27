# Sourced by the SARC tools. Zone definitions for dev/1.5 (see sarc/README.md).
SARC_BASE=${SARC_BASE:-origin/release/1.5}      # upstream release the branch forks from
SARC_RELEASE_BRANCH=${SARC_RELEASE_BRANCH:-sarc/1.5}
SARC_RELEASE_ZONE=(
  backends/vulkan/runtime/graph/ops/glsl/sarc/
  backends/vulkan/runtime/graph/ops/impl/sarc/
)
SARC_DEV_ZONE=(
  backends/vulkan/runtime/graph/ops/glsl/sarc_dev/
  backends/vulkan/runtime/graph/ops/impl/sarc_dev/
  backends/vulkan/test/sarc_dev/
  sarc/
  openspec/
)
# Twin template wrappers: identical from "#version" on.
SARC_TWINS=(
  "backends/vulkan/runtime/graph/ops/glsl/sarc/sarc_linear_q4gsw_coopmat.glsl backends/vulkan/runtime/graph/ops/glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.glsl"
  "backends/vulkan/runtime/graph/ops/glsl/sarc/sarc_linear_dq8ca_zpg.glsl backends/vulkan/runtime/graph/ops/glsl/sarc_dev/sarc_linear_dq8ca_zpg_sweep.glsl"
  "backends/vulkan/runtime/graph/ops/glsl/sarc/sarc_linear_dq8ca_zpgtr.glsl backends/vulkan/runtime/graph/ops/glsl/sarc_dev/sarc_linear_dq8ca_zpgtr_sweep.glsl"
)
sarc_in_zone() { # sarc_in_zone <path> <zone...>
  local p=$1; shift
  local z; for z in "$@"; do [[ $p == "$z"* ]] && return 0; done; return 1
}
sarc_hooks() { grep -v '^\s*#' "$SARC_ROOT/sarc/HOOKS" | awk 'NF {print $1}'; }
