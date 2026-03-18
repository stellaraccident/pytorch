#pragma once

// IREE compiler wrapper: MLIR → VMFB compilation with two backends.
//
// PYRE_IREE_COMPILE     — Path to iree-compile CLI (debug/dev)
// PYRE_IREE_COMPILER_LIB — Path to libIREECompiler.so (dlopen, production)
//
// CompilerOutput owns the resulting VMFB with alignment guarantees.
// CompilerResult carries either a CompilerOutput or an error message.
// compile() returns a shared_future<CompilerResult>.

#include <iree/io/file_contents.h>

#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace at::pyre {

// Owns compiled VMFB output with page-aligned data.
// Data remains valid for the lifetime of this object.
class CompilerOutput {
 public:
  virtual ~CompilerOutput() = default;
  virtual const uint8_t* data() const = 0;
  virtual size_t size() const = 0;

  iree_const_byte_span_t span() const {
    return {data(), static_cast<iree_host_size_t>(size())};
  }

  // Transfer ownership of this output to an iree_allocator_t.
  // The allocator's free calls delete on this CompilerOutput.
  // After calling, do NOT delete the CompilerOutput separately.
  iree_allocator_t transferDeallocator();

 private:
  static iree_status_t deallocator_ctl(
      void* self, iree_allocator_command_t command,
      const void* params, void** inout_ptr);
};

// CompilerOutput backed by iree_io_file_contents_t (page-aligned).
class FileContentsOutput final : public CompilerOutput {
 public:
  static std::unique_ptr<FileContentsOutput> fromBytes(
      const uint8_t* bytes, size_t len, iree_allocator_t alloc);
  static std::unique_ptr<FileContentsOutput> fromFile(
      const std::string& path, iree_allocator_t alloc);

  ~FileContentsOutput() override;
  const uint8_t* data() const override;
  size_t size() const override;

 private:
  explicit FileContentsOutput(iree_io_file_contents_t* contents);
  iree_io_file_contents_t* contents_;
};

// Result of a compilation attempt.
// shared_ptr so CompilerResult is copyable (required by shared_future::get).
struct CompilerResult {
  std::shared_ptr<CompilerOutput> output;
  std::string error_message;

  bool ok() const { return output != nullptr; }
};

class TORCH_PYRE_API PyreKernelCompiler {
 public:
  static bool initialize();
  static bool isAvailable();

  // Compile MLIR text to VMFB. Takes ownership of mlir_asm (move).
  // Returns a future that resolves to CompilerResult.
  static std::shared_future<CompilerResult> compile(
      std::string mlir_asm,
      std::vector<std::string> flags);

  // Convenience: compile and block. TORCH_CHECK on failure.
  static std::shared_ptr<CompilerOutput> compileSync(
      std::string mlir_asm,
      std::vector<std::string> flags);

 private:
  static std::once_flag init_flag_;
  static bool available_;
  static bool use_cli_;
  static std::string cli_path_;
  static bool compiler_lib_loaded_;

  static void doInitialize();
  static CompilerResult compileCLI(
      const std::string& mlir_asm, const std::vector<std::string>& flags);
  static CompilerResult compileCAPI(
      const std::string& mlir_asm, const std::vector<std::string>& flags);
};

} // namespace at::pyre
