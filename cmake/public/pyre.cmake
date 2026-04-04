# Pyre: ROCm backend for PyTorch, replacing the hipified CUDA path.
# Consumes the standalone pyre runtime package for HAL execution and graph
# compilation.
#
# pyre runtime package location — set via -Dpyre_DIR=..., -DPYRE_DIR=...,
# CMAKE_PREFIX_PATH, or env var PYRE_DIR.
if(NOT DEFINED pyre_DIR AND DEFINED PYRE_DIR)
  set(pyre_DIR "${PYRE_DIR}")
elseif(NOT DEFINED pyre_DIR AND DEFINED ENV{PYRE_DIR})
  set(pyre_DIR "$ENV{PYRE_DIR}")
endif()

find_package(pyre CONFIG REQUIRED)

set(PYTORCH_FOUND_PYRE ON)
message(STATUS "Pyre: using pyre package at ${pyre_DIR}")

# Collect variables for downstream targets.
set(Caffe2_PYRE_DEPENDENCY_LIBS pyre::pyre)
set(Caffe2_PYRE_INCLUDE "")
