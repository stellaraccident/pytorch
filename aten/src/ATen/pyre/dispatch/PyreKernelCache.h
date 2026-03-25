#pragma once

// Three-tier kernel cache with shared_future dedup.
//
// On cache miss, the first thread to claim a key becomes the compiler.
// Other threads hitting the same key wait on the shared_future instead
// of compiling redundantly.

#include <ATen/pyre/dispatch/PyreAbiConfig.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreVMContext.h>

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace at::pyre {

class PyreKernelCache {
 public:
  struct LookupResult {
    std::shared_future<CachedKernel*> future;
    bool is_compiler;  // true = you must compile and call fulfill/fail
  };

  // Returns a future that resolves to the kernel.
  // On hit: future is already ready (immediate .get()).
  // On miss: first caller gets is_compiler=true and must compile.
  //          Other callers get is_compiler=false and wait.
  LookupResult lookupOrClaim(const std::string& cache_key,
                              const std::string& func_name,
                              const AbiConfig& abi = AbiConfig::kEnvelope);

  // Called by the compiler thread to fulfill the promise.
  void fulfill(const std::string& cache_key,
               std::shared_ptr<CompilerOutput> vmfb,
               const std::string& func_name,
               const AbiConfig& abi = AbiConfig::kEnvelope);

  // Called on compilation failure to unblock waiters.
  void fail(const std::string& cache_key, std::exception_ptr ex);

  // Disk cache operations.
  std::shared_ptr<CompilerOutput> loadFromDisk(const std::string& cache_key);
  void saveToDisk(const std::string& cache_key, const CompilerOutput& output);

  static PyreKernelCache& get();

 private:
  PyreKernelCache() = default;
  std::string diskCachePath(const std::string& cache_key) const;
  std::string cacheDir() const;

  std::mutex mutex_;
  // Completed kernels.
  std::unordered_map<std::string, CachedKernel> ready_;
  // In-flight compilations.
  std::unordered_map<std::string,
      std::shared_ptr<std::promise<CachedKernel*>>> pending_;
};

} // namespace at::pyre
