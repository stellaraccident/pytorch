#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace at::pyre {

std::once_flag PyreKernelCompiler::init_flag_;
bool PyreKernelCompiler::available_ = false;
bool PyreKernelCompiler::use_cli_ = false;
std::string PyreKernelCompiler::cli_path_;
bool PyreKernelCompiler::compiler_lib_loaded_ = false;

void PyreKernelCompiler::doInitialize() {
  PYRE_LOG(INFO) << "initializing IREE compiler\n";

  // 1. Check PYRE_IREE_COMPILE env var (CLI path, takes precedence).
  if (const char* cli = std::getenv("PYRE_IREE_COMPILE")) {
    cli_path_ = cli;
    // Verify the binary exists.
    if (std::filesystem::exists(cli_path_)) {
      use_cli_ = true;
      available_ = true;
      LOG(INFO) << "pyre: using IREE compiler CLI at " << cli_path_;
      return;
    }
    LOG(WARNING) << "pyre: PYRE_IREE_COMPILE set to " << cli_path_
                 << " but file not found";
  }

  // 2. Check PYRE_IREE_COMPILER_LIB env var (dlopen path).
  if (const char* lib = std::getenv("PYRE_IREE_COMPILER_LIB")) {
    if (ireeCompilerLoadLibrary(lib)) {
      ireeCompilerGlobalInitialize();
      compiler_lib_loaded_ = true;
      available_ = true;
      LOG(INFO) << "pyre: loaded IREE compiler library from " << lib;
      return;
    }
    LOG(WARNING) << "pyre: PYRE_IREE_COMPILER_LIB set to " << lib
                 << " but failed to load";
  }

  // 3. Try standard library paths via loader.
  // Search common paths where libIREECompiler.so might be installed.
  for (const char* path : {
           "libIREECompiler.so",
           "libIREECompiler.so.0",
       }) {
    if (ireeCompilerLoadLibrary(path)) {
      ireeCompilerGlobalInitialize();
      compiler_lib_loaded_ = true;
      available_ = true;
      LOG(INFO) << "pyre: loaded IREE compiler library from system path";
      return;
    }
  }

  LOG(INFO) << "pyre: IREE compiler not found. Set PYRE_IREE_COMPILE or "
               "PYRE_IREE_COMPILER_LIB to enable kernel compilation.";
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

std::vector<uint8_t> PyreKernelCompiler::compileCLI(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  // Write MLIR to a temp file.
  auto tmp_dir = std::filesystem::temp_directory_path();
  auto mlir_path = tmp_dir / "pyre_kernel.mlir";
  auto vmfb_path = tmp_dir / "pyre_kernel.vmfb";

  {
    std::ofstream f(mlir_path);
    TORCH_CHECK(f.is_open(), "pyre: failed to create temp file ", mlir_path);
    f << mlir_asm;
  }

  PYRE_LOG(INFO) << "compiling kernel via CLI: " << mlir_path << "\n";
  PYRE_LOG(DEBUG) << "MLIR source (" << mlir_asm.size() << " bytes):\n"
                  << mlir_asm << "\n";

  // Build command line.
  std::ostringstream cmd;
  cmd << cli_path_;
  for (const auto& flag : flags) {
    cmd << " " << flag;
  }
  cmd << " -o " << vmfb_path << " " << mlir_path;
  cmd << " 2>&1";

  // Run iree-compile.
  std::array<char, 4096> buf{};
  std::string output;
  FILE* pipe = popen(cmd.str().c_str(), "r");
  TORCH_CHECK(pipe, "pyre: failed to run iree-compile");

  while (fgets(buf.data(), buf.size(), pipe)) {
    output += buf.data();
  }
  int status = pclose(pipe);

  // Clean up input.
  std::filesystem::remove(mlir_path);

  TORCH_CHECK(
      status == 0,
      "pyre: iree-compile failed (exit code ", status, "):\n", output);

  // Read the VMFB output.
  std::ifstream vmfb_file(vmfb_path, std::ios::binary);
  TORCH_CHECK(
      vmfb_file.is_open(), "pyre: iree-compile did not produce output at ",
      vmfb_path);

  std::vector<uint8_t> vmfb(
      (std::istreambuf_iterator<char>(vmfb_file)),
      std::istreambuf_iterator<char>());

  // Clean up output.
  std::filesystem::remove(vmfb_path);

  TORCH_CHECK(!vmfb.empty(), "pyre: iree-compile produced empty VMFB");
  PYRE_LOG(INFO) << "CLI compilation succeeded, VMFB size: "
                 << vmfb.size() << " bytes\n";
  return vmfb;
}

// -------------------------------------------------------------------------- //
// C API backend
// -------------------------------------------------------------------------- //

std::vector<uint8_t> PyreKernelCompiler::compileCAPI(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  PYRE_LOG(INFO) << "compiling kernel via C API (" << mlir_asm.size()
                 << " bytes MLIR)\n";
  PYRE_LOG(DEBUG) << "MLIR source:\n" << mlir_asm << "\n";

  // Create session and set flags.
  iree_compiler_session_t* session = ireeCompilerSessionCreate();

  if (!flags.empty()) {
    std::vector<const char*> argv;
    argv.reserve(flags.size());
    for (const auto& f : flags) {
      argv.push_back(f.c_str());
    }
    iree_compiler_error_t* err = ireeCompilerSessionSetFlags(
        session, static_cast<int>(argv.size()), argv.data());
    if (err) {
      std::string msg = ireeCompilerErrorGetMessage(err);
      ireeCompilerErrorDestroy(err);
      ireeCompilerSessionDestroy(session);
      TORCH_CHECK(false, "pyre: failed to set compiler flags: ", msg);
    }
  }

  // Create invocation.
  iree_compiler_invocation_t* inv = ireeCompilerInvocationCreate(session);

  // Collect diagnostics.
  std::string diagnostics;
  ireeCompilerInvocationEnableCallbackDiagnostics(
      inv, 0,
      [](enum iree_compiler_diagnostic_severity_t severity,
         const char* message, size_t messageSize, void* userData) {
        auto* diag = static_cast<std::string*>(userData);
        if (severity >= IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR) {
          if (!diag->empty()) *diag += "\n";
          diag->append(message, messageSize);
        }
      },
      &diagnostics);

  // Parse source from memory buffer.
  iree_compiler_source_t* source = nullptr;
  iree_compiler_error_t* src_err = ireeCompilerSourceWrapBuffer(
      session, "pyre_kernel.mlir",
      mlir_asm.c_str(), mlir_asm.size() + 1,
      /*isNullTerminated=*/true, &source);
  if (src_err) {
    std::string msg = ireeCompilerErrorGetMessage(src_err);
    ireeCompilerErrorDestroy(src_err);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    TORCH_CHECK(false, "pyre: failed to create compiler source: ", msg);
  }

  bool parsed = ireeCompilerInvocationParseSource(inv, source);
  if (!parsed) {
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    TORCH_CHECK(false, "pyre: failed to parse MLIR:\n", diagnostics);
  }

  // Run the standard compilation pipeline.
  bool compiled = ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD);
  if (!compiled) {
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    TORCH_CHECK(false, "pyre: IREE compilation failed:\n", diagnostics);
  }

  // Output VMFB to memory buffer.
  iree_compiler_output_t* output = nullptr;
  iree_compiler_error_t* out_err = ireeCompilerOutputOpenMembuffer(&output);
  if (out_err) {
    std::string msg = ireeCompilerErrorGetMessage(out_err);
    ireeCompilerErrorDestroy(out_err);
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    TORCH_CHECK(false, "pyre: failed to create output buffer: ", msg);
  }

  iree_compiler_error_t* vm_err =
      ireeCompilerInvocationOutputVMBytecode(inv, output);
  if (vm_err) {
    std::string msg = ireeCompilerErrorGetMessage(vm_err);
    ireeCompilerErrorDestroy(vm_err);
    ireeCompilerOutputDestroy(output);
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    TORCH_CHECK(false, "pyre: failed to emit VMFB: ", msg);
  }

  // Map the output memory.
  void* contents = nullptr;
  uint64_t size = 0;
  iree_compiler_error_t* map_err =
      ireeCompilerOutputMapMemory(output, &contents, &size);
  if (map_err) {
    std::string msg = ireeCompilerErrorGetMessage(map_err);
    ireeCompilerErrorDestroy(map_err);
    ireeCompilerOutputDestroy(output);
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    TORCH_CHECK(false, "pyre: failed to map output memory: ", msg);
  }

  // Copy to vector.
  std::vector<uint8_t> vmfb(
      static_cast<uint8_t*>(contents),
      static_cast<uint8_t*>(contents) + size);

  // Cleanup.
  ireeCompilerOutputDestroy(output);
  ireeCompilerSourceDestroy(source);
  ireeCompilerInvocationDestroy(inv);
  ireeCompilerSessionDestroy(session);

  return vmfb;
}

// -------------------------------------------------------------------------- //
// Public API
// -------------------------------------------------------------------------- //

std::shared_future<std::vector<uint8_t>> PyreKernelCompiler::compile(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  TORCH_CHECK(
      isAvailable(),
      "pyre: IREE compiler not available. Set PYRE_IREE_COMPILE or "
      "PYRE_IREE_COMPILER_LIB environment variable.");

  // Epic 1: synchronous compilation wrapped in a future.
  // Future: submit to background thread pool.
  std::promise<std::vector<uint8_t>> promise;
  try {
    std::vector<uint8_t> vmfb;
    if (use_cli_) {
      vmfb = compileCLI(mlir_asm, flags);
    } else {
      vmfb = compileCAPI(mlir_asm, flags);
    }
    promise.set_value(std::move(vmfb));
  } catch (...) {
    promise.set_exception(std::current_exception());
  }
  return promise.get_future().share();
}

std::vector<uint8_t> PyreKernelCompiler::compileSync(
    const std::string& mlir_asm,
    const std::vector<std::string>& flags) {
  return compile(mlir_asm, flags).get();
}

} // namespace at::pyre
