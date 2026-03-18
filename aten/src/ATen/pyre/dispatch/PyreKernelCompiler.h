#pragma once

// C++ wrapper around the IREE compiler for on-demand MLIR → VMFB compilation.
//
// Two backends, selected by environment:
//   PYRE_IREE_COMPILE     — Path to iree-compile CLI (debug/dev)
//   PYRE_IREE_COMPILER_LIB — Path to libIREECompiler.so (dlopen, production)
//
// compile() returns a shared_future following the FBGEMM CodeCache pattern.
// Hard failure on compile error (TORCH_CHECK).

#include <c10/macros/Export.h>

#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace at::pyre {

class TORCH_PYRE_API PyreKernelCompiler {
 public:
  // Initialize compiler. Checks env vars, loads library if available.
  // Returns false if no compiler found (JIT disabled).
  static bool initialize();
  static bool isAvailable();

  // Compile MLIR text to VMFB bytes. Returns a shared_future.
  // Compilation runs synchronously for now (Epic 1); the future interface
  // supports background compilation in the future.
  static std::shared_future<std::vector<uint8_t>> compile(
      const std::string& mlir_asm,
      const std::vector<std::string>& flags);

  // Convenience: compile and block on result.
  static std::vector<uint8_t> compileSync(
      const std::string& mlir_asm,
      const std::vector<std::string>& flags);

 private:
  static std::once_flag init_flag_;
  static bool available_;
  static bool use_cli_;
  static std::string cli_path_;
  static bool compiler_lib_loaded_;

  static void doInitialize();

  // CLI backend: runs iree-compile as subprocess.
  static std::vector<uint8_t> compileCLI(
      const std::string& mlir_asm,
      const std::vector<std::string>& flags);

  // C API backend: uses dlopen'd libIREECompiler.so.
  static std::vector<uint8_t> compileCAPI(
      const std::string& mlir_asm,
      const std::vector<std::string>& flags);
};

} // namespace at::pyre
