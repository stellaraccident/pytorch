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
  return cacheDir() + "/" + cache_key + ".vmfb";
}

PyreKernelCache::LookupResult PyreKernelCache::lookupOrClaim(
    const std::string& cache_key,
    const std::string& func_name,
    const AbiConfig& abi) {
  std::lock_guard<std::mutex> lock(mutex_);

  // 1. Memory hit.
  auto it = ready_.find(cache_key);
  if (it != ready_.end()) {
    PYRE_LOG(TRACE) << "cache hit (memory): " << cache_key << "\n";
    std::promise<CachedKernel*> p;
    p.set_value(&it->second);
    return {p.get_future().share(), false};
  }

  // 2. Already being compiled by another thread — wait on its future.
  auto pit = pending_.find(cache_key);
  if (pit != pending_.end()) {
    PYRE_LOG(INFO) << "cache pending (waiting): " << cache_key << "\n";
    return {pit->second->get_future().share(), false};
  }

  // 3. Disk hit — load, store in ready_, return immediately.
  //    (loadFromDisk doesn't need the lock, but store does — keep it simple.)
  auto vmfb = loadFromDisk(cache_key);
  if (vmfb) {
    PYRE_LOG(INFO) << "cache hit (disk): " << cache_key << "\n";
    auto kernel = loadKernel(std::move(vmfb), func_name, abi);
    auto [inserted_it, ok] = ready_.emplace(cache_key, std::move(kernel));
    std::promise<CachedKernel*> p;
    p.set_value(&inserted_it->second);
    return {p.get_future().share(), false};
  }

  // 4. True miss — claim this key for compilation.
  PYRE_LOG(INFO) << "cache MISS: " << cache_key << ", compiling\n";
  auto promise = std::make_shared<std::promise<CachedKernel*>>();
  pending_.emplace(cache_key, promise);
  return {promise->get_future().share(), true};
}

void PyreKernelCache::fulfill(
    const std::string& cache_key,
    std::shared_ptr<CompilerOutput> vmfb,
    const std::string& func_name,
    const AbiConfig& abi) {
  std::lock_guard<std::mutex> lock(mutex_);

  saveToDisk(cache_key, *vmfb);

  auto kernel = loadKernel(std::move(vmfb), func_name, abi);
  auto [it, ok] = ready_.emplace(cache_key, std::move(kernel));

  auto pit = pending_.find(cache_key);
  if (pit != pending_.end()) {
    pit->second->set_value(&it->second);
    pending_.erase(pit);
  }
}

void PyreKernelCache::fail(
    const std::string& cache_key,
    std::exception_ptr ex) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto pit = pending_.find(cache_key);
  if (pit != pending_.end()) {
    pit->second->set_exception(ex);
    pending_.erase(pit);
  }
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
