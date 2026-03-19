#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreStringSplicer.h>
#include <c10/util/hash.h>

#include <sstream>

// Generated template strings (build step: tools/embed_templates.py).
// Not checked in — regenerate with:
//   python tools/embed_templates.py aten/src/ATen/pyre/dispatch/templates/*.mlir \
//       -o aten/src/ATen/pyre/dispatch/PyreKernelTemplates.inc
#include <ATen/pyre/dispatch/PyreKernelTemplates.inc>

namespace at::pyre {

std::string contentHashCacheKey(
    const char* aten_name,
    const char* template_sha1,
    const std::vector<std::pair<std::string, std::string>>& substitutions,
    c10::ArrayRef<std::string> compiler_flags) {
  std::string to_hash;
  to_hash += template_sha1;
  to_hash += '\0';
  for (const auto& [key, value] : substitutions) {
    to_hash += key;
    to_hash += '=';
    to_hash += value;
    to_hash += '\0';
  }
  for (const auto& flag : compiler_flags) {
    to_hash += flag;
    to_hash += '\0';
  }
  c10::sha1 hasher(to_hash);
  return hasher.str();
}

// Stringify a shape with size-1 dims literal, others dynamic "?".
// This is what torch-mlir needs for broadcast.
static std::string broadcastAwareShapeStr(c10::ArrayRef<int64_t> sizes) {
  std::string s;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) s += ",";
    s += (sizes[i] == 1) ? "1" : "?";
  }
  return s;
}

// Concrete shape string with actual sizes — needed for permuted inputs
// so the compiler can track dimension reordering.
static std::string concreteShapeStr(c10::ArrayRef<int64_t> sizes) {
  std::string s;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) s += ",";
    s += std::to_string(sizes[i]);
  }
  return s;
}

// Check if any adapter in the list is kPermute.
static bool hasPermutedAdapter(const std::vector<ArgAdapter>& adapters) {
  for (const auto& a : adapters)
    if (a.kind == ArgAdapter::kPermute) return true;
  return false;
}

// Emit torch.aten.permute MLIR: builds a prim.ListConstruct for the perm,
// then calls torch.aten.permute.
static std::string emitPermuteLines(
    const std::string& dst_name,
    const std::string& src_name,
    const std::vector<int64_t>& perm,
    const std::string& src_type,
    const std::string& dst_type) {
  std::ostringstream ss;
  // Emit constants for each permutation index.
  for (size_t i = 0; i < perm.size(); ++i) {
    ss << "    %p_" << dst_name << "_" << i
       << " = torch.constant.int " << perm[i] << "\n";
  }
  // Build the list.
  ss << "    %perm_" << dst_name << " = torch.prim.ListConstruct ";
  for (size_t i = 0; i < perm.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << "%p_" << dst_name << "_" << i;
  }
  ss << " : (";
  for (size_t i = 0; i < perm.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << "!torch.int";
  }
  ss << ") -> !torch.list<int>\n";
  // Permute.
  ss << "    %" << dst_name << " = torch.aten.permute %" << src_name
     << ", %perm_" << dst_name << " : " << src_type
     << ", !torch.list<int> -> " << dst_type;
  return ss.str();
}

// Compute inverse permutation.
static std::vector<int64_t> inversePerm(const std::vector<int64_t>& perm) {
  std::vector<int64_t> inv(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    inv[perm[i]] = static_cast<int64_t>(i);
  return inv;
}

// Compute logical shape from physical shape and adapter permutation.
// The adapter's perm is the stride-sorted order. The inverse perm
// applied to the physical shape gives the logical shape.
static std::vector<int64_t> logicalShape(
    c10::ArrayRef<int64_t> physical_shape,
    const std::vector<int64_t>& perm) {
  auto inv = inversePerm(perm);
  std::vector<int64_t> logical(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    logical[i] = physical_shape[inv[i]];
  return logical;
}

// Build binary MLIR with permute adapters when needed.
static ExpandedKernel expandBinaryWithPermute(
    const std::string& func_name,
    const std::string& torch_op,
    const std::string& extra_args,
    const std::string& extra_arg_decls,
    const std::string& extra_arg_types,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_phys_shape,
    c10::ArrayRef<int64_t> rhs_phys_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  auto out_str = broadcastAwareShapeStr(out_shape);

  bool lhs_permuted = adapters.size() > 0
      && adapters[0].kind == ArgAdapter::kPermute;
  bool rhs_permuted = adapters.size() > 1
      && adapters[1].kind == ArgAdapter::kPermute;

  // For permuted inputs, use concrete shapes so the compiler can track
  // dimension reordering through torch.aten.permute.
  auto lhs_phys = lhs_permuted
      ? concreteShapeStr(lhs_phys_shape) : broadcastAwareShapeStr(lhs_phys_shape);
  auto rhs_phys = rhs_permuted
      ? concreteShapeStr(rhs_phys_shape) : broadcastAwareShapeStr(rhs_phys_shape);

  auto lhs_log_shape = lhs_permuted
      ? logicalShape(lhs_phys_shape, adapters[0].permutation)
      : std::vector<int64_t>(lhs_phys_shape.begin(), lhs_phys_shape.end());
  auto rhs_log_shape = rhs_permuted
      ? logicalShape(rhs_phys_shape, adapters[1].permutation)
      : std::vector<int64_t>(rhs_phys_shape.begin(), rhs_phys_shape.end());
  auto lhs_log = lhs_permuted
      ? concreteShapeStr(lhs_log_shape) : broadcastAwareShapeStr(lhs_log_shape);
  auto rhs_log = rhs_permuted
      ? concreteShapeStr(rhs_log_shape) : broadcastAwareShapeStr(rhs_log_shape);

  std::string lhs_phys_type = "!torch.vtensor<[" + lhs_phys + "], " + elt + ">";
  std::string rhs_phys_type = "!torch.vtensor<[" + rhs_phys + "], " + elt + ">";
  std::string lhs_log_type = "!torch.vtensor<[" + lhs_log + "], " + elt + ">";
  std::string rhs_log_type = "!torch.vtensor<[" + rhs_log + "], " + elt + ">";
  std::string out_type = "!torch.vtensor<[" + out_str + "], " + elt + ">";
  std::string out_ttype = "!torch.tensor<[" + out_str + "], " + elt + ">";

  std::ostringstream ss;
  ss << "module @module {\n"
     << "  func.func @" << func_name << "(\n"
     << "      %out_: " << out_ttype << ",\n"
     << "      %lhs_phys: " << lhs_phys_type << ",\n"
     << "      %rhs_phys: " << rhs_phys_type << "\n"
     << "  ) attributes {torch.assume_strict_symbolic_shapes} {\n";

  // Emit permute ops to recover logical shapes.
  // The MLIR permute goes physical → logical, which is the inverse
  // of the adapter's perm (adapter perm is logical → physical order).
  if (lhs_permuted) {
    ss << emitPermuteLines("lhs", "lhs_phys",
           inversePerm(adapters[0].permutation),
           lhs_phys_type, lhs_log_type) << "\n";
  }
  if (rhs_permuted) {
    ss << emitPermuteLines("rhs", "rhs_phys",
           inversePerm(adapters[1].permutation),
           rhs_phys_type, rhs_log_type) << "\n";
  }

  // For non-permuted args, use the phys name directly (they're the same).
  std::string lhs_name = lhs_permuted ? "lhs" : "lhs_phys";
  std::string rhs_name = rhs_permuted ? "rhs" : "rhs_phys";
  std::string lhs_use_type = lhs_permuted ? lhs_log_type : lhs_phys_type;
  std::string rhs_use_type = rhs_permuted ? rhs_log_type : rhs_phys_type;

  if (!extra_arg_decls.empty())
    ss << extra_arg_decls << "\n";

  ss << "    %result = " << torch_op << " %" << lhs_name << ", %" << rhs_name
     << extra_args << " : " << lhs_use_type << ", " << rhs_use_type
     << extra_arg_types << " -> " << out_type << "\n"
     << "    torch.overwrite.tensor.contents %result overwrites %out_ : "
     << out_type << ", " << out_ttype << "\n"
     << "    return\n"
     << "  }\n"
     << "}\n";

  // For permuted paths, build substitutions from all the shape/op info
  // so the content hash captures the full MLIR variation.
  std::vector<std::pair<std::string, std::string>> subs = {
      {"func_name", func_name}, {"torch_op", torch_op},
      {"lhs_phys", lhs_phys}, {"rhs_phys", rhs_phys},
      {"lhs_log", lhs_log}, {"rhs_log", rhs_log},
      {"out_shape", out_str}, {"extra_args", extra_args},
      {"extra_arg_decls", extra_arg_decls},
  };
  for (size_t i = 0; i < adapters.size(); ++i) {
    if (adapters[i].kind == ArgAdapter::kPermute) {
      std::string p;
      for (auto v : adapters[i].permutation) p += std::to_string(v) + ",";
      subs.push_back({"perm_" + std::to_string(i), p});
    }
  }
  return {ss.str(), kTemplate_elementwise_binary_sha1, std::move(subs)};
}

ExpandedKernel expandBinaryTemplate(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters) {
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

  if (hasPermutedAdapter(adapters)) {
    return expandBinaryWithPermute(
        func_name, torch_op, extra_args, extra_arg_decls,
        extra_arg_types, dtype, lhs_shape, rhs_shape, out_shape, adapters);
  }

  std::vector<std::pair<std::string, std::string>> vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"lhs_shape", broadcastAwareShapeStr(lhs_shape)},
      {"rhs_shape", broadcastAwareShapeStr(rhs_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"torch_op", torch_op}, {"extra_args", extra_args},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_arg_types", extra_arg_types},
  };
  return {pyreSplice(kTemplate_elementwise_binary, vars),
          kTemplate_elementwise_binary_sha1, std::move(vars)};
}

ExpandedKernel expandBinaryAlphaTemplate(
    const std::string& func_name,
    const std::string& alpha_add_op,
    const std::string& /*alpha_mul_op*/,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters) {
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

  if (hasPermutedAdapter(adapters)) {
    return expandBinaryWithPermute(
        func_name, torch_op, ", %alpha", alpha_decl.str(),
        ", !torch.int", dtype, lhs_shape, rhs_shape, out_shape, adapters);
  }

  std::vector<std::pair<std::string, std::string>> vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"lhs_shape", broadcastAwareShapeStr(lhs_shape)},
      {"rhs_shape", broadcastAwareShapeStr(rhs_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"torch_op", torch_op}, {"extra_args", ", %alpha"},
      {"extra_arg_decls", alpha_decl.str()},
      {"extra_arg_types", ", !torch.int"},
  };
  return {pyreSplice(kTemplate_elementwise_binary, vars),
          kTemplate_elementwise_binary_sha1, std::move(vars)};
}

ExpandedKernel expandUnaryTemplate(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string shape;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (i > 0) shape += ",";
    shape += "?";
  }

  if (adapter.kind == ArgAdapter::kPermute) {
    auto phys_str = concreteShapeStr(input_shape);
    auto log_shape = logicalShape(input_shape, adapter.permutation);
    auto log_str = concreteShapeStr(log_shape);

    std::string phys_type = "!torch.vtensor<[" + phys_str + "], " + elt + ">";
    std::string log_type = "!torch.vtensor<[" + log_str + "], " + elt + ">";
    std::string out_type = "!torch.vtensor<[" + log_str + "], " + elt + ">";
    std::string out_ttype = "!torch.tensor<[" + log_str + "], " + elt + ">";

    std::ostringstream ss;
    ss << "module @module {\n"
       << "  func.func @" << func_name << "(\n"
       << "      %out_: " << out_ttype << ",\n"
       << "      %input_phys: " << phys_type << "\n"
       << "  ) attributes {torch.assume_strict_symbolic_shapes} {\n"
       << emitPermuteLines("input", "input_phys",
              inversePerm(adapter.permutation), phys_type, log_type) << "\n"
       << "    %result = " << scalar_op << " %input : "
       << log_type << " -> " << out_type << "\n"
       << "    torch.overwrite.tensor.contents %result overwrites %out_ : "
       << out_type << ", " << out_ttype << "\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    std::vector<std::pair<std::string, std::string>> subs = {
        {"element_type", elt}, {"func_name", func_name},
        {"input_phys_shape", phys_str}, {"input_log_shape", log_str},
        {"torch_op", scalar_op},
    };
    std::string p;
    for (auto v : adapter.permutation) p += std::to_string(v) + ",";
    subs.push_back({"perm", p});
    return {ss.str(), kTemplate_elementwise_unary_sha1, std::move(subs)};
  }

  std::vector<std::pair<std::string, std::string>> vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", shape}, {"out_shape", shape},
      {"torch_op", scalar_op},
  };
  return {pyreSplice(kTemplate_elementwise_unary, vars),
          kTemplate_elementwise_unary_sha1, std::move(vars)};
}

ExpandedKernel expandAddmmTemplate(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::vector<std::pair<std::string, std::string>> vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_shape", broadcastAwareShapeStr(mat2_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
  return {pyreSplice(kTemplate_addmm, vars),
          kTemplate_addmm_sha1, std::move(vars)};
}

ExpandedKernel expandAddmmTransposedTemplate(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string mat2_t = broadcastAwareShapeStr(
      {mat2_orig_shape[1], mat2_orig_shape[0]});
  std::vector<std::pair<std::string, std::string>> vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_orig_shape", broadcastAwareShapeStr(mat2_orig_shape)},
      {"mat2_t_shape", mat2_t},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
  return {pyreSplice(kTemplate_addmm_transposed, vars),
          kTemplate_addmm_transposed_sha1, std::move(vars)};
}

} // namespace at::pyre
