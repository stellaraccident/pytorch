#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <c10/pyre/impl/PyreRuntime.h>
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
  if (const char* dir = std::getenv("PYRE_CACHE_DIR"))
    return std::string(dir) + "/kernels";
  if (const char* home = std::getenv("HOME"))
    return std::string(home) + "/.cache/pyre/kernels";
  return "/tmp/pyre/kernels";
}

std::string PyreKernelCache::diskCachePath(const std::string& cache_key) const {
  // cache_key is a 40-char SHA1 hex digest — filesystem-safe as-is.
  return cacheDir() + "/" + cache_key + ".vmfb";
}

CachedKernel* PyreKernelCache::lookup(
    const std::string& cache_key,
    const std::string& func_name,
    bool native_abi) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
      PYRE_LOG(TRACE) << "cache hit (memory): " << cache_key << "\n";
      return &it->second;
    }
  }

  PYRE_LOG(TRACE) << "cache miss (memory): " << cache_key << "\n";

  auto vmfb = loadFromDisk(cache_key);
  if (vmfb) {
    PYRE_LOG(INFO) << "cache hit (disk): " << cache_key << "\n";
    return store(cache_key, func_name, std::move(vmfb), native_abi);
  }

  return nullptr;
}

CachedKernel* PyreKernelCache::store(
    const std::string& cache_key,
    const std::string& func_name,
    std::shared_ptr<CompilerOutput> vmfb,
    bool native_abi) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(cache_key);
  if (it != cache_.end()) return &it->second;

  // Save to disk before loading (best-effort).
  saveToDisk(cache_key, *vmfb);

  auto kernel = loadKernel(std::move(vmfb), func_name, native_abi);
  auto [inserted_it, ok] = cache_.emplace(cache_key, std::move(kernel));
  return &inserted_it->second;
}

std::shared_ptr<CompilerOutput> PyreKernelCache::loadFromDisk(
    const std::string& cache_key) {
  auto path = diskCachePath(cache_key);
  if (!std::filesystem::exists(path)) return nullptr;

  try {
    auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();
    return FileContentsOutput::fromFile(path, alloc);
  } catch (...) {
    return nullptr;
  }
}

void PyreKernelCache::saveToDisk(
    const std::string& cache_key,
    const CompilerOutput& output) {
  try {
    auto path = diskCachePath(cache_key);
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
      f.write(reinterpret_cast<const char*>(output.data()),
              static_cast<std::streamsize>(output.size()));
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "pyre: disk cache write failed: " << e.what();
  }
}

} // namespace at::pyre
