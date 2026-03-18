#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreStringSplicer.h>

#include <sstream>

// Generated template strings (build step: tools/embed_templates.py).
// Not checked in — regenerate with:
//   python tools/embed_templates.py aten/src/ATen/pyre/dispatch/templates/*.mlir \
//       -o aten/src/ATen/pyre/dispatch/PyreKernelTemplates.inc
#include <ATen/pyre/dispatch/PyreKernelTemplates.inc>

namespace at::pyre {

// Build shape string for torch.vtensor types.
// Broadcast dims (size 1) are static "1"; others are dynamic "?".
// Stringify a shape with size-1 dims literal, others dynamic "?".
// This is what torch-mlir needs for broadcast: the compiler must see
// the "1" to generate broadcast code.
static std::string broadcastAwareShapeStr(c10::ArrayRef<int64_t> sizes) {
  std::string s;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) s += ",";
    s += (sizes[i] == 1) ? "1" : "?";
  }
  return s;
}

std::string expandBinaryTemplate(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& /*adapters*/) {
  std::string elt = scalarTypeToTorchMlir(dtype);

  std::string torch_op, extra_args, extra_arg_decls, extra_arg_types;
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
  } else if (linalg_op == "mm") {
    torch_op = "torch.aten.mm";
  } else {
    TORCH_CHECK(false, "pyre: unknown binary op: ", linalg_op);
  }

  return pyreSplice(kTemplate_elementwise_binary, {
      {"element_type", elt}, {"func_name", func_name},
      {"lhs_shape", broadcastAwareShapeStr(lhs_shape)},
      {"rhs_shape", broadcastAwareShapeStr(rhs_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"torch_op", torch_op}, {"extra_args", extra_args},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_arg_types", extra_arg_types},
  });
}

std::string expandBinaryAlphaTemplate(
    const std::string& func_name,
    const std::string& alpha_add_op,
    const std::string& /*alpha_mul_op*/,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& /*adapters*/) {
  std::string elt = scalarTypeToTorchMlir(dtype);

  std::string torch_op = (alpha_add_op.find("sub") != std::string::npos)
      ? "torch.aten.sub.Tensor" : "torch.aten.add.Tensor";

  std::ostringstream alpha_decl;
  if (alpha_value == static_cast<int64_t>(alpha_value))
    alpha_decl << "    %alpha = torch.constant.int "
               << static_cast<int64_t>(alpha_value);
  else
    alpha_decl << "    %alpha = torch.constant.float "
               << std::fixed << alpha_value;

  return pyreSplice(kTemplate_elementwise_binary, {
      {"element_type", elt}, {"func_name", func_name},
      {"lhs_shape", broadcastAwareShapeStr(lhs_shape)},
      {"rhs_shape", broadcastAwareShapeStr(rhs_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"torch_op", torch_op}, {"extra_args", ", %alpha"},
      {"extra_arg_decls", alpha_decl.str()},
      {"extra_arg_types", ", !torch.int"},
  });
}

std::string expandUnaryTemplate(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> /*input_shape*/,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& /*adapter*/) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  // Unary: all dynamic, no broadcast specialization needed.
  std::string shape;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (i > 0) shape += ",";
    shape += "?";
  }

  return pyreSplice(kTemplate_elementwise_unary, {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", shape}, {"out_shape", shape},
      {"torch_op", scalar_op},
  });
}

std::string expandAddmmTemplate(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  return pyreSplice(kTemplate_addmm, {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_shape", broadcastAwareShapeStr(mat2_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  });
}

std::string expandAddmmTransposedTemplate(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  // mat2_t_shape is mat2_orig_shape with dims swapped.
  std::string mat2_t = broadcastAwareShapeStr(
      {mat2_orig_shape[1], mat2_orig_shape[0]});
  return pyreSplice(kTemplate_addmm_transposed, {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_orig_shape", broadcastAwareShapeStr(mat2_orig_shape)},
      {"mat2_t_shape", mat2_t},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  });
}

} // namespace at::pyre
