#include <ATen/pyre/dispatch/PyreKernelCompiler.h>

#include <pyre_compiler_cxx.h>

#include <c10/util/Exception.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace at::pyre {
namespace {

class CompilerHandleOutput final : public CompilerOutput {
 public:
  explicit CompilerHandleOutput(
      ::pyre::compiler::compiler_output_ptr output)
      : output_(std::move(output)) {}

  const uint8_t* data() const override {
    return pyre_compiler_output_data(output_.get());
  }

  size_t size() const override {
    return pyre_compiler_output_size(output_.get());
  }

  void loadInto(pyre_device_t device,
                pyre_module_t* module) const override {
    PYRE_CHECK_OK(pyre_module_load_compiler_output(
        device, output_.get(), module));
  }

 private:
  ::pyre::compiler::compiler_output_ptr output_;
};

std::unique_ptr<::pyre::compiler::PyreGraphCompiler>& graphCompiler() {
  static std::unique_ptr<::pyre::compiler::PyreGraphCompiler> compiler;
  return compiler;
}

} // namespace

// -------------------------------------------------------------------------- //
// FileContentsOutput
// -------------------------------------------------------------------------- //

FileContentsOutput::FileContentsOutput(
    pyre_host_allocator_t allocator, uint8_t* data, size_t size)
    : allocator_(allocator), data_(data), size_(size) {}

FileContentsOutput::~FileContentsOutput() {
  pyre_host_allocator_free_aligned(allocator_, data_);
}

const uint8_t* FileContentsOutput::data() const {
  return data_;
}

size_t FileContentsOutput::size() const {
  return size_;
}

void FileContentsOutput::loadInto(
    pyre_device_t device, pyre_module_t* module) const {
  PYRE_CHECK_OK(pyre_module_load_vmfb(device, data_, size_, module));
}

std::unique_ptr<FileContentsOutput> FileContentsOutput::fromBytes(
    const uint8_t* bytes, size_t len, pyre_host_allocator_t alloc) {
  constexpr size_t kAlignment = 4096;
  uint8_t* data = nullptr;
  PYRE_CHECK_OK(pyre_host_allocator_malloc_aligned(
      alloc, len + 1, kAlignment, /*offset=*/0,
      reinterpret_cast<void**>(&data)));
  if (len > 0) {
    std::memcpy(data, bytes, len);
  }
  data[len] = 0;
  return std::unique_ptr<FileContentsOutput>(
      new FileContentsOutput(alloc, data, len));
}

std::unique_ptr<FileContentsOutput> FileContentsOutput::fromFile(
    const std::string& path, pyre_host_allocator_t alloc) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(file.is_open(), "pyre: failed to open VMFB file: ", path);
  std::streamsize size = file.tellg();
  TORCH_CHECK(size >= 0, "pyre: failed to stat VMFB file: ", path);
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0) {
    file.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    TORCH_CHECK(file, "pyre: failed to read VMFB file: ", path);
  }
  return fromBytes(bytes.data(), bytes.size(), alloc);
}

// -------------------------------------------------------------------------- //
// Compiler statics
// -------------------------------------------------------------------------- //

std::once_flag PyreGraphCompiler::init_flag_;
bool PyreGraphCompiler::available_ = false;

void PyreGraphCompiler::doInitialize() {
  PYRE_LOG(INFO) << "initializing Pyre graph compiler\n";
  try {
    graphCompiler() =
        std::make_unique<::pyre::compiler::PyreGraphCompiler>(
            PYRE_COMPILER_BACKEND_AUTO);
    available_ = true;
    PYRE_LOG(INFO) << "using graph compiler backend "
                   << graphCompiler()->backend() << " via libpyre\n";
    return;
  } catch (const std::exception& e) {
    PYRE_LOG(INFO) << "Pyre graph compiler not available: " << e.what()
                   << "\n";
  }

  PYRE_LOG(INFO) << "Pyre graph compiler not found\n";
}

bool PyreGraphCompiler::initialize() {
  std::call_once(init_flag_, doInitialize);
  return available_;
}

bool PyreGraphCompiler::isAvailable() {
  initialize();
  return available_;
}

CompilerResult PyreGraphCompiler::compileGraph(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  try {
    auto output = graphCompiler()->compileMlir(mlir_asm, flags);
    auto result = std::make_shared<CompilerHandleOutput>(
        std::move(output));
    PYRE_LOG(INFO) << "graph compilation succeeded, VMFB "
                   << result->size() << " bytes\n";
    return {std::move(result), {}};
  } catch (const std::exception& e) {
    return {{}, e.what()};
  }
}

std::shared_future<CompilerResult> PyreGraphCompiler::compile(
    const std::string& mlir_asm,
    c10::ArrayRef<std::string> flags) {
  TORCH_CHECK(isAvailable(),
      "pyre: IREE compiler not available. "
      "Set PYRE_IREE_COMPILE or PYRE_IREE_COMPILER_CLI.");

  std::vector<std::string> flag_vec(flags.begin(), flags.end());

  std::promise<CompilerResult> promise;
  promise.set_value(compileGraph(mlir_asm, flag_vec));
  return promise.get_future().share();
}

std::shared_ptr<CompilerOutput> PyreGraphCompiler::compileSync(
    const std::string& mlir_asm,
    c10::ArrayRef<std::string> flags) {
  auto result = compile(mlir_asm, flags).get();
  TORCH_CHECK(result.ok(), "pyre: compilation failed: ", result.error_message);
  return result.output;
}

} // namespace at::pyre
