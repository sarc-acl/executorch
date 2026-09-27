# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SARC development zone: test and benchmark binaries. Standalone project built
# against an installed vulkan_backend, like ../custom_ops (see
# sarc/tools/build.sh). Not part of a release.

cmake_minimum_required(VERSION 3.19)
project(sarc_dev_tests)

if(ANDROID)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)

set(VULKAN_CXX_FLAGS "-fexceptions" "-DUSE_VULKAN_WRAPPER" "-DUSE_VULKAN_VOLK")

if(TARGET vulkan_backend)
  if(NOT EXECUTORCH_ROOT)
    set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
  endif()

  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

  set(VULKAN_THIRD_PARTY_PATH ${EXECUTORCH_ROOT}/backends/vulkan/third-party)
  set(COMMON_INCLUDES
      ${EXECUTORCH_ROOT}/.. ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include
      ${VULKAN_THIRD_PARTY_PATH}/volk
      ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator
  )

  executorch_target_link_options_shared_lib(vulkan_backend)

  # Llama-shape linear/SDPA microbenchmark with correctness and
  # production-diff checks. utils.{h,cpp} is a copy of ../custom_ops/utils.*
  # with the sampled-reference (NaN-skip) extension.
  add_executable(
    test_llama_microbench
    ${CMAKE_CURRENT_SOURCE_DIR}/test_llama_microbench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  )
  target_include_directories(
    test_llama_microbench PRIVATE ${COMMON_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(
    test_llama_microbench PRIVATE vulkan_backend executorch_core
  )
  target_compile_options(test_llama_microbench PRIVATE ${VULKAN_CXX_FLAGS})
  set_property(TARGET test_llama_microbench PROPERTY CXX_STANDARD 17)
endif()
