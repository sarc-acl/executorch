# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SARC development zone: logits_probe, a runner that writes the next-token
# logits of a list of token prompts (one prefill call each). Built against the
# install tree of an llm-preset build (sarc/tools/build.sh --llama), with the
# link set of examples/models/llama; not part of a release. See
# openspec/changes/sarc-1.5-780m-prefill-refine/tools/build_probe.sh.

cmake_minimum_required(VERSION 3.19)
project(sarc_dev_logits_probe)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch extension_module_static extension_tensor)
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib optimized_kernels
       portable_kernels cpublas eigen_blas
  )
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()
if(TARGET quantized_ops_lib)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  get_target_property(_quantized_imported quantized_ops_lib IMPORTED)
  if(NOT _quantized_imported)
    executorch_target_link_options_shared_lib(quantized_ops_lib)
  endif()
endif()
if(TARGET custom_ops)
  executorch_target_link_options_shared_lib(custom_ops)
  list(APPEND link_libraries custom_ops)
endif()
if(TARGET torchao_ops_executorch)
  executorch_target_link_options_shared_lib(torchao_ops_executorch)
  list(APPEND link_libraries torchao_ops_executorch)
endif()
if(TARGET extension_threadpool)
  list(APPEND link_libraries extension_threadpool pthreadpool cpuinfo)
endif()
if(TARGET xnnpack_backend)
  list(APPEND link_libraries xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()
list(APPEND link_libraries vulkan_backend)
executorch_target_link_options_shared_lib(vulkan_backend)

add_executable(logits_probe logits_probe.cpp)
target_include_directories(logits_probe PRIVATE ${EXECUTORCH_ROOT}/..)
target_link_libraries(logits_probe PRIVATE ${link_libraries})
target_compile_options(logits_probe PRIVATE -Wno-deprecated-declarations -fPIC)
set_property(TARGET logits_probe PROPERTY CXX_STANDARD 17)
