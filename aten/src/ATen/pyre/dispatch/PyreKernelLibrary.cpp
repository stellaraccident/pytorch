#include <ATen/pyre/dispatch/PyreKernelLibrary.h>
#include <ATen/pyre/dispatch/PyreKernelTemplates.inc>
#include <ATen/pyre/dispatch/PyreStringSplicer.h>

#include <sstream>

namespace at::pyre {

// Map PyTorch ScalarType to torch-mlir element type string.
// torch-mlir uses "f32", "f64", "si32", "si64" (signed integers).
static const char* torchMlirElementType(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float: return "f32";
    case c10::ScalarType::Double: return "f64";
    case c10::ScalarType::Half: return "f16";
    case c10::ScalarType::BFloat16: return "bf16";
    case c10::ScalarType::Int: return "si32";
    case c10::ScalarType::Long: return "si64";
    case c10::ScalarType::Short: return "si16";
    case c10::ScalarType::Byte: return "ui8";
    case c10::ScalarType::Char: return "si8";
    case c10::ScalarType::Bool: return "i1";
    default:
      TORCH_CHECK(false, "pyre: unsupported dtype: ", c10::toString(dtype));
  }
}

// Build shape string with commas for torch.vtensor (e.g., "?,?" for rank 2).
static std::string dynamicShapeComma(int64_t rank) {
  if (rank == 0) return "";
  std::string s = "?";
  for (int64_t i = 1; i < rank; ++i) {
    s += ",?";
  }
  return s;
}

std::string dynamicShapeStr(int64_t rank) {
  if (rank == 0) return "";
  std::string s = "?";
  for (int64_t i = 1; i < rank; ++i) {
    s += "x?";
  }
  return s;
}

std::string dimVarsStr(int64_t rank) {
  if (rank == 0) return "";
  std::ostringstream ss;
  for (int64_t i = 0; i < rank; ++i) {
    if (i > 0) ss << ", ";
    ss << "d" << i;
  }
  return ss.str();
}

std::string parallelTypesStr(int64_t rank) {
  if (rank == 0) return "";
  std::ostringstream ss;
  for (int64_t i = 0; i < rank; ++i) {
    if (i > 0) ss << ", ";
    ss << "\"parallel\"";
  }
  return ss.str();
}

std::string expandBinaryTemplate(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters) {
  int64_t rank = static_cast<int64_t>(out_shape.size());
  std::string shape = dynamicShapeComma(rank);
  std::string elt = torchMlirElementType(dtype);

  // Map linalg op name to torch dialect op.
  std::string torch_op;
  std::string extra_args;
  std::string extra_arg_decls;
  std::string extra_arg_types;
  if (linalg_op == "add") {
    torch_op = "torch.aten.add.Tensor";
    extra_arg_decls = "    %alpha = torch.constant.int 1";
    extra_args = ", %alpha";
    extra_arg_types = ", !torch.int";
  } else if (linalg_op == "sub") {
    torch_op = "torch.aten.sub.Tensor";
    extra_arg_decls = "    %alpha = torch.constant.int 1";
    extra_args = ", %alpha";
    extra_arg_types = ", !torch.int";
  } else if (linalg_op == "mul") {
    torch_op = "torch.aten.mul.Tensor";
  } else if (linalg_op == "div") {
    torch_op = "torch.aten.div.Tensor";
  } else {
    TORCH_CHECK(false, "pyre: unknown binary op: ", linalg_op);
  }

  return pyreSplice(kTemplate_elementwise_binary, {
      {"element_type", elt},
      {"func_name", func_name},
      {"lhs_shape", shape},
      {"rhs_shape", shape},
      {"out_shape", shape},
      {"torch_op", torch_op},
      {"extra_args", extra_args},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_arg_types", extra_arg_types},
  });
}

std::string expandBinaryAlphaTemplate(
    const std::string& func_name,
    const std::string& alpha_add_op,
    const std::string& alpha_mul_op,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters) {
  int64_t rank = static_cast<int64_t>(out_shape.size());
  std::string shape = dynamicShapeComma(rank);
  std::string elt = torchMlirElementType(dtype);

  // Use torch.aten.add.Tensor with non-unit alpha.
  // alpha_add_op tells us whether it's add or sub.
  std::string torch_op = (alpha_add_op.find("sub") != std::string::npos)
      ? "torch.aten.sub.Tensor" : "torch.aten.add.Tensor";

  // Format alpha as torch constant.
  std::ostringstream alpha_decl;
  if (alpha_value == static_cast<int64_t>(alpha_value)) {
    alpha_decl << "    %alpha = torch.constant.int "
               << static_cast<int64_t>(alpha_value);
  } else {
    alpha_decl << "    %alpha = torch.constant.float " << std::fixed
               << alpha_value;
  }

  return pyreSplice(kTemplate_elementwise_binary, {
      {"element_type", elt},
      {"func_name", func_name},
      {"lhs_shape", shape},
      {"rhs_shape", shape},
      {"out_shape", shape},
      {"torch_op", torch_op},
      {"extra_args", ", %alpha"},
      {"extra_arg_decls", alpha_decl.str()},
      {"extra_arg_types", ", !torch.int"},
  });
}

std::string expandUnaryTemplate(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter) {
  int64_t rank = static_cast<int64_t>(out_shape.size());
  std::string shape = dynamicShapeComma(rank);
  std::string elt = torchMlirElementType(dtype);

  // scalar_op is the torch dialect op name (torch.aten.neg, etc.)
  return pyreSplice(kTemplate_elementwise_unary, {
      {"element_type", elt},
      {"func_name", func_name},
      {"input_shape", shape},
      {"out_shape", shape},
      {"torch_op", scalar_op},
  });
}

} // namespace at::pyre
