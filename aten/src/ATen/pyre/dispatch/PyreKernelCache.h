#pragma once

// Three-tier kernel cache following FBGEMM CodeCache pattern.
//
// Tier 1: In-memory — shared_future<CachedKernel> per key. Concurrent
//         requests for the same key share one future (no duplicate compiles).
// Tier 2: User disk — VMFB files at $PYRE_CACHE_DIR/kernels/<device>/.
// Tier 3: System — pre-compiled kernel pack (future, not yet implemented).
//
// Lookup order: memory → disk → system → compile.
// See epic1_kernel_dispatch.md §4.8.

#include <ATen/pyre/dispatch/PyreVMContext.h>

#include <future>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::pyre {

class PyreKernelCache {
 public:
  // Look up a cached kernel. Returns nullptr on cache miss.
  CachedKernel* lookup(const std::string& cache_key,
                        const std::string& func_name);

  // Store a compiled kernel in the cache. Returns pointer to stored kernel.
  CachedKernel* store(const std::string& cache_key,
                       const std::string& func_name,
                       const std::vector<uint8_t>& vmfb);

  // Try to load from disk cache. Returns empty vector on miss.
  std::vector<uint8_t> loadFromDisk(const std::string& cache_key);

  // Save to disk cache.
  void saveToDisk(const std::string& cache_key,
                  const std::vector<uint8_t>& vmfb);

  static PyreKernelCache& get();

 private:
  PyreKernelCache() = default;

  std::string diskCachePath(const std::string& cache_key) const;
  std::string cacheDir() const;

  std::shared_timed_mutex mutex_;
  std::unordered_map<std::string, CachedKernel> cache_;
};

} // namespace at::pyre
