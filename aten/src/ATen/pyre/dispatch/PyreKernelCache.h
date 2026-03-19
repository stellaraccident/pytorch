#pragma once

// Three-tier kernel cache: in-memory → disk → system → compile.
// Uses std::mutex (matching PyTorch convention).

#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreVMContext.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace at::pyre {

class PyreKernelCache {
 public:
  // Returns cached kernel or nullptr on miss.
  // If native_abi is true, resolves bare function name (native IREE).
  CachedKernel* lookup(const std::string& cache_key,
                        const std::string& func_name,
                        bool native_abi = false);

  // Store a compiled kernel. Returns pointer to stored entry.
  CachedKernel* store(const std::string& cache_key,
                       const std::string& func_name,
                       std::shared_ptr<CompilerOutput> vmfb,
                       bool native_abi = false);

  // Disk cache operations.
  std::shared_ptr<CompilerOutput> loadFromDisk(const std::string& cache_key);
  void saveToDisk(const std::string& cache_key, const CompilerOutput& output);

  static PyreKernelCache& get();

 private:
  PyreKernelCache() = default;
  std::string diskCachePath(const std::string& cache_key) const;
  std::string cacheDir() const;

  std::mutex mutex_;
  std::unordered_map<std::string, CachedKernel> cache_;
};

} // namespace at::pyre
