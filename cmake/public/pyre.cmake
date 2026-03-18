# Pyre: ROCm backend for PyTorch, replacing the hipified CUDA path.
# Uses IREE runtime for HAL (device, memory, execution).
#
# Takes a source dependency on IREE runtime via add_subdirectory,
# following the same pattern as Fusilli.

# IREE source directory — set via -DPYRE_IREE_SOURCE_DIR=... or env var.
if(NOT DEFINED PYRE_IREE_SOURCE_DIR AND DEFINED ENV{PYRE_IREE_SOURCE_DIR})
  set(PYRE_IREE_SOURCE_DIR "$ENV{PYRE_IREE_SOURCE_DIR}")
endif()

if(NOT DEFINED PYRE_IREE_SOURCE_DIR OR NOT EXISTS "${PYRE_IREE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "USE_PYRE=ON requires PYRE_IREE_SOURCE_DIR to point to an IREE source tree. "
    "Set -DPYRE_IREE_SOURCE_DIR=/path/to/iree or export PYRE_IREE_SOURCE_DIR=/path/to/iree")
endif()

message(STATUS "Pyre: using IREE source at ${PYRE_IREE_SOURCE_DIR}")

# Configure IREE for runtime-only build (no compiler, no tests).
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_VISIBILITY_HIDDEN OFF)

# CPU drivers only for Phase 0.
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_DRIVER_LOCAL_TASK ON)

# Force IREE to build static libs even though PyTorch uses BUILD_SHARED_LIBS=ON.
# This is critical — IREE's many small libraries would otherwise produce
# dozens of .so files that aren't installed to torch/lib.
set(_pyre_saved_build_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${PYRE_IREE_SOURCE_DIR} pyre_iree SYSTEM EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_pyre_saved_build_shared})

set(PYTORCH_FOUND_PYRE ON)

# Collect variables for downstream targets.
set(Caffe2_PYRE_DEPENDENCY_LIBS iree_runtime_unified)
set(Caffe2_PYRE_INCLUDE ${PYRE_IREE_SOURCE_DIR}/runtime/src)
