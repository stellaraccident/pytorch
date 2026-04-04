#pragma once

// Pyre graph compiler wrapper: MLIR -> VMFB compilation via libpyre.
//
// PYRE_IREE_COMPILE      — Path to libIREECompiler.so (dlopen, production)
// PYRE_IREE_COMPILER_CLI — Path to iree-compile CLI (debug/dev fallback)
//
// CompilerOutput owns the resulting VMFB bytes.
// CompilerResult carries either a CompilerOutput or an error message.
// compile() returns a shared_future<CompilerResult>.

#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/util/ArrayRef.h>

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
  virtual void loadInto(pyre_device_t device, pyre_module_t* module) const = 0;
};

// CompilerOutput backed by pyre host allocator memory (page-aligned).
class FileContentsOutput final : public CompilerOutput {
 public:
  static std::unique_ptr<FileContentsOutput> fromBytes(
      const uint8_t* bytes, size_t len, pyre_host_allocator_t alloc);
  static std::unique_ptr<FileContentsOutput> fromFile(
      const std::string& path, pyre_host_allocator_t alloc);

  ~FileContentsOutput() override;
  const uint8_t* data() const override;
  size_t size() const override;
  void loadInto(pyre_device_t device, pyre_module_t* module) const override;

 private:
  FileContentsOutput(
      pyre_host_allocator_t allocator, uint8_t* data, size_t size);
  pyre_host_allocator_t allocator_;
  uint8_t* data_;
  size_t size_;
};

// Result of a compilation attempt.
// shared_ptr so CompilerResult is copyable (required by shared_future::get).
struct CompilerResult {
  std::shared_ptr<CompilerOutput> output;
  std::string error_message;

  bool ok() const { return output != nullptr; }
};

class TORCH_PYRE_API PyreGraphCompiler {
 public:
  static bool initialize();
  static bool isAvailable();

  // Compile MLIR text to VMFB. Returns a future (currently synchronous).
  static std::shared_future<CompilerResult> compile(
      const std::string& mlir_asm,
      c10::ArrayRef<std::string> flags);

  // Convenience: compile and block. TORCH_CHECK on failure.
  static std::shared_ptr<CompilerOutput> compileSync(
      const std::string& mlir_asm,
      c10::ArrayRef<std::string> flags);

 private:
  static std::once_flag init_flag_;
  static bool available_;

  static void doInitialize();
  static CompilerResult compileGraph(
      const std::string& mlir_asm, const std::vector<std::string>& flags);
};

using PyreKernelCompiler = PyreGraphCompiler;

} // namespace at::pyre
