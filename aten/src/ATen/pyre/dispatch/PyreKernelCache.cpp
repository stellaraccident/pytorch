#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/util/Logging.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace at::pyre {

PyreKernelCache& PyreKernelCache::get() {
  static PyreKernelCache instance;
  return instance;
}

std::string PyreKernelCache::cacheDir() const {
  if (const char* dir = std::getenv("PYRE_CACHE_DIR")) {
    return std::string(dir) + "/kernels";
  }
  if (const char* home = std::getenv("HOME")) {
    return std::string(home) + "/.cache/pyre/kernels";
  }
  return "/tmp/pyre/kernels";
}

std::string PyreKernelCache::diskCachePath(const std::string& cache_key) const {
  return cacheDir() + "/" + cache_key + ".vmfb";
}

CachedKernel* PyreKernelCache::lookup(
    const std::string& cache_key,
    const std::string& func_name) {
  // Fast path: shared lock.
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
      PYRE_LOG(TRACE) << "in-memory cache hit: " << cache_key
                      << " func_ordinal=" << it->second.function.ordinal
                      << " module=" << (it->second.module.get() ? "valid" : "null")
                      << " context=" << (it->second.context.get() ? "valid" : "null")
                      << "\n";
      return &it->second;
    }
  }

  PYRE_LOG(TRACE) << "cache lookup: " << cache_key << "\n";

  // Try disk cache.
  auto vmfb = loadFromDisk(cache_key);
  if (!vmfb.empty()) {
    PYRE_LOG(INFO) << "disk cache hit: " << cache_key << "\n";
    return store(cache_key, func_name, vmfb);
  }

  return nullptr;
}

CachedKernel* PyreKernelCache::store(
    const std::string& cache_key,
    const std::string& func_name,
    const std::vector<uint8_t>& vmfb) {
  // Double-check under write lock.
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  auto it = cache_.find(cache_key);
  if (it != cache_.end()) {
    return &it->second;
  }

  auto kernel = loadKernel(vmfb, func_name);
  auto [inserted, _] = cache_.emplace(cache_key, std::move(kernel));

  // Save to disk cache asynchronously (best-effort).
  saveToDisk(cache_key, vmfb);

  return &inserted->second;
}

std::vector<uint8_t> PyreKernelCache::loadFromDisk(
    const std::string& cache_key) {
  auto path = diskCachePath(cache_key);
  if (!std::filesystem::exists(path)) {
    return {};
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return {};

  return std::vector<uint8_t>(
      (std::istreambuf_iterator<char>(f)),
      std::istreambuf_iterator<char>());
}

void PyreKernelCache::saveToDisk(
    const std::string& cache_key,
    const std::vector<uint8_t>& vmfb) {
  try {
    auto path = diskCachePath(cache_key);
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
      f.write(reinterpret_cast<const char*>(vmfb.data()),
              static_cast<std::streamsize>(vmfb.size()));
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "pyre: failed to save kernel to disk cache: " << e.what();
  }
}

} // namespace at::pyre
