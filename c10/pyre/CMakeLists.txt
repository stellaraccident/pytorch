# c10/pyre — Pyre device layer for DeviceType::HIP
#
# Wraps IREE HAL for device management, memory allocation, and stream
# execution. This replaces c10/hip for pyre builds.

include(../../cmake/public/utils.cmake)

file(GLOB C10_PYRE_SRCS
  *.cpp
  impl/*.cpp
)

file(GLOB_RECURSE C10_PYRE_HEADERS *.h)

add_library(c10_pyre ${C10_PYRE_SRCS} ${C10_PYRE_HEADERS})
torch_compile_options(c10_pyre)

target_compile_options(c10_pyre PRIVATE "-DC10_PYRE_BUILD_MAIN_LIB")
target_compile_definitions(c10_pyre PRIVATE USE_PYRE)

if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_pyre PRIVATE "-fvisibility=hidden")
endif()

target_link_libraries(c10_pyre PUBLIC ${C10_LIB} pyre::pyre)

target_include_directories(
  c10_pyre PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>)

install(TARGETS c10_pyre EXPORT Caffe2Targets DESTINATION lib)
set(C10_PYRE_LIB c10_pyre)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION include
  FILES_MATCHING PATTERN "*.h")
