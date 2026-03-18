#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/util/Exception.h>

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace at::pyre {

// -------------------------------------------------------------------------- //
// CompilerOutput
// -------------------------------------------------------------------------- //

iree_status_t CompilerOutput::deallocator_ctl(
    void* self, iree_allocator_command_t command,
    const void* /*params*/, void** /*inout_ptr*/) {
  if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    delete static_cast<CompilerOutput*>(self);
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "CompilerOutput deallocator supports only free");
}

iree_allocator_t CompilerOutput::transferDeallocator() {
  return {/*.self=*/this, /*.ctl=*/deallocator_ctl};
}

// -------------------------------------------------------------------------- //
// FileContentsOutput
// -------------------------------------------------------------------------- //

FileContentsOutput::FileContentsOutput(iree_io_file_contents_t* contents)
    : contents_(contents) {}

FileContentsOutput::~FileContentsOutput() {
  if (contents_) iree_io_file_contents_free(contents_);
}

const uint8_t* FileContentsOutput::data() const {
  return contents_->const_buffer.data;
}

size_t FileContentsOutput::size() const {
  return contents_->const_buffer.data_length;
}

std::unique_ptr<FileContentsOutput> FileContentsOutput::fromBytes(
    const uint8_t* bytes, size_t len, iree_allocator_t alloc) {
  constexpr size_t kAlignment = 4096;
  size_t header = sizeof(iree_io_file_contents_t);
  size_t data_offset = (header + kAlignment - 1) & ~(kAlignment - 1);
  size_t total = data_offset + len + 1;

  void* block = nullptr;
  PYRE_CHECK_OK(iree_allocator_malloc(alloc, total, &block));

  auto* contents = static_cast<iree_io_file_contents_t*>(block);
  contents->allocator = alloc;
  contents->buffer.data = static_cast<uint8_t*>(block) + data_offset;
  contents->buffer.data_length = len;
  contents->mapping = nullptr;
  std::memcpy(contents->buffer.data, bytes, len);
  contents->buffer.data[len] = 0;

  return std::unique_ptr<FileContentsOutput>(new FileContentsOutput(contents));
}

std::unique_ptr<FileContentsOutput> FileContentsOutput::fromFile(
    const std::string& path, iree_allocator_t alloc) {
  iree_io_file_contents_t* contents = nullptr;
  iree_string_view_t pv = {path.c_str(), static_cast<iree_host_size_t>(path.size())};
  PYRE_CHECK_OK(iree_io_file_contents_read(pv, alloc, &contents));
  return std::unique_ptr<FileContentsOutput>(new FileContentsOutput(contents));
}

// -------------------------------------------------------------------------- //
// Compiler statics
// -------------------------------------------------------------------------- //

std::once_flag PyreKernelCompiler::init_flag_;
bool PyreKernelCompiler::available_ = false;
bool PyreKernelCompiler::use_cli_ = false;
std::string PyreKernelCompiler::cli_path_;
bool PyreKernelCompiler::compiler_lib_loaded_ = false;

void PyreKernelCompiler::doInitialize() {
  PYRE_LOG(INFO) << "initializing IREE compiler\n";

  if (const char* cli = std::getenv("PYRE_IREE_COMPILE")) {
    cli_path_ = cli;
    if (std::filesystem::exists(cli_path_)) {
      use_cli_ = true;
      available_ = true;
      PYRE_LOG(INFO) << "using IREE compiler CLI at " << cli_path_ << "\n";
      return;
    }
    PYRE_LOG(WARN) << "PYRE_IREE_COMPILE=" << cli_path_
                   << " but file not found\n";
  }

  if (const char* lib = std::getenv("PYRE_IREE_COMPILER_LIB")) {
    if (ireeCompilerLoadLibrary(lib)) {
      ireeCompilerGlobalInitialize();
      compiler_lib_loaded_ = true;
      available_ = true;
      PYRE_LOG(INFO) << "loaded IREE compiler library from " << lib << "\n";
      return;
    }
  }

  for (const char* path : {"libIREECompiler.so", "libIREECompiler.so.0"}) {
    if (ireeCompilerLoadLibrary(path)) {
      ireeCompilerGlobalInitialize();
      compiler_lib_loaded_ = true;
      available_ = true;
      PYRE_LOG(INFO) << "loaded IREE compiler from system path\n";
      return;
    }
  }

  PYRE_LOG(INFO) << "IREE compiler not found\n";
}

bool PyreKernelCompiler::initialize() {
  std::call_once(init_flag_, doInitialize);
  return available_;
}

bool PyreKernelCompiler::isAvailable() {
  initialize();
  return available_;
}

// -------------------------------------------------------------------------- //
// CLI backend
// -------------------------------------------------------------------------- //

CompilerResult PyreKernelCompiler::compileCLI(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  // Unique temp files.
  auto pid = std::to_string(getpid());
  auto tid = std::to_string(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  auto tmp = std::filesystem::temp_directory_path();
  auto mlir_path = tmp / ("pyre_" + pid + "_" + tid + ".mlir");
  auto vmfb_path = tmp / ("pyre_" + pid + "_" + tid + ".vmfb");

  {
    std::ofstream f(mlir_path);
    if (!f.is_open())
      return {{}, "failed to create temp file " + mlir_path.string()};
    f << mlir_asm;
  }

  PYRE_LOG(INFO) << "compiling via CLI: " << mlir_path << "\n";
  PYRE_LOG(DEBUG) << "MLIR (" << mlir_asm.size() << " bytes):\n"
                  << mlir_asm << "\n";

  // Build command. Read stderr via pipe, not shell redirect.
  std::string cmd = cli_path_;
  for (const auto& flag : flags) cmd += " " + flag;
  cmd += " -o " + vmfb_path.string() + " " + mlir_path.string();
  cmd += " 2>&1";

  std::array<char, 4096> buf{};
  std::string output;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::filesystem::remove(mlir_path);
    return {{}, "failed to run iree-compile"};
  }
  while (fgets(buf.data(), buf.size(), pipe)) output += buf.data();
  int status = pclose(pipe);

  std::filesystem::remove(mlir_path);

  if (status != 0) {
    std::filesystem::remove(vmfb_path);
    return {{}, "iree-compile failed (exit " + std::to_string(status) +
                    "):\n" + output};
  }

  // Read VMFB with proper alignment.
  auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();
  auto result = FileContentsOutput::fromFile(vmfb_path.string(), alloc);
  std::filesystem::remove(vmfb_path);

  PYRE_LOG(INFO) << "CLI compilation succeeded, VMFB "
                 << result->size() << " bytes\n";
  return {std::move(result), {}};
}

// -------------------------------------------------------------------------- //
// C API backend — RAII state management
// -------------------------------------------------------------------------- //

namespace {
// Holds all CAPI objects for one compilation. Destructor cleans up on any path.
struct CompilerState {
  iree_compiler_session_t* session = nullptr;
  iree_compiler_invocation_t* inv = nullptr;
  iree_compiler_source_t* source = nullptr;
  iree_compiler_output_t* output = nullptr;
  iree_compiler_error_t* error = nullptr;
  std::string diagnostics;

  ~CompilerState() {
    if (output) ireeCompilerOutputDestroy(output);
    if (source) ireeCompilerSourceDestroy(source);
    if (inv) ireeCompilerInvocationDestroy(inv);
    if (session) ireeCompilerSessionDestroy(session);
    if (error) ireeCompilerErrorDestroy(error);
  }

  // Set an error, returning a CompilerResult with the message.
  CompilerResult fail(const char* context) {
    std::string msg = context;
    if (error) {
      msg += ": ";
      msg += ireeCompilerErrorGetMessage(error);
      ireeCompilerErrorDestroy(error);
      error = nullptr;
    }
    if (!diagnostics.empty()) {
      msg += "\n";
      msg += diagnostics;
    }
    return {{}, std::move(msg)};
  }
};
} // namespace

CompilerResult PyreKernelCompiler::compileCAPI(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  PYRE_LOG(INFO) << "compiling via C API (" << mlir_asm.size() << " bytes)\n";
  PYRE_LOG(DEBUG) << "MLIR:\n" << mlir_asm << "\n";

  CompilerState st;
  st.session = ireeCompilerSessionCreate();

  if (!flags.empty()) {
    std::vector<const char*> argv;
    argv.reserve(flags.size());
    for (const auto& f : flags) argv.push_back(f.c_str());
    st.error = ireeCompilerSessionSetFlags(
        st.session, static_cast<int>(argv.size()), argv.data());
    if (st.error) return st.fail("failed to set compiler flags");
  }

  st.inv = ireeCompilerInvocationCreate(st.session);
  ireeCompilerInvocationEnableCallbackDiagnostics(
      st.inv, 0,
      [](enum iree_compiler_diagnostic_severity_t severity,
         const char* message, size_t messageSize, void* ud) {
        auto* diag = static_cast<std::string*>(ud);
        if (severity >= IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR) {
          if (!diag->empty()) *diag += "\n";
          diag->append(message, messageSize);
        }
      },
      &st.diagnostics);

  st.error = ireeCompilerSourceWrapBuffer(
      st.session, "pyre_kernel.mlir",
      mlir_asm.c_str(), mlir_asm.size() + 1,
      /*isNullTerminated=*/true, &st.source);
  if (st.error) return st.fail("failed to create compiler source");

  if (!ireeCompilerInvocationParseSource(st.inv, st.source))
    return st.fail("failed to parse MLIR");

  if (!ireeCompilerInvocationPipeline(st.inv, IREE_COMPILER_PIPELINE_STD))
    return st.fail("IREE compilation failed");

  st.error = ireeCompilerOutputOpenMembuffer(&st.output);
  if (st.error) return st.fail("failed to create output buffer");

  st.error = ireeCompilerInvocationOutputVMBytecode(st.inv, st.output);
  if (st.error) return st.fail("failed to emit VMFB");

  void* contents = nullptr;
  uint64_t sz = 0;
  st.error = ireeCompilerOutputMapMemory(st.output, &contents, &sz);
  if (st.error) return st.fail("failed to map output memory");

  // Copy to page-aligned buffer (the mmap'd output lifetime is tied to
  // the compiler output handle which we're about to destroy).
  auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();
  auto result = FileContentsOutput::fromBytes(
      static_cast<const uint8_t*>(contents), static_cast<size_t>(sz), alloc);

  PYRE_LOG(INFO) << "C API compilation succeeded, VMFB "
                 << result->size() << " bytes\n";
  return {std::move(result), {}};
}

// -------------------------------------------------------------------------- //
// Public API
// -------------------------------------------------------------------------- //

std::shared_future<CompilerResult> PyreKernelCompiler::compile(
    std::string mlir_asm,
    std::vector<std::string> flags) {
  TORCH_CHECK(isAvailable(),
      "pyre: IREE compiler not available. "
      "Set PYRE_IREE_COMPILE or PYRE_IREE_COMPILER_LIB.");

  // Epic 1: synchronous, wrapped in a future for API compatibility.
  std::promise<CompilerResult> promise;
  if (use_cli_) {
    promise.set_value(compileCLI(mlir_asm, flags));
  } else {
    promise.set_value(compileCAPI(mlir_asm, flags));
  }
  return promise.get_future().share();
}

std::shared_ptr<CompilerOutput> PyreKernelCompiler::compileSync(
    std::string mlir_asm,
    std::vector<std::string> flags) {
  auto result = compile(std::move(mlir_asm), std::move(flags)).get();
  TORCH_CHECK(result.ok(), "pyre: compilation failed: ", result.error_message);
  return result.output;
}

} // namespace at::pyre
