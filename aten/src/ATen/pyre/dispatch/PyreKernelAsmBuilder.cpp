#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreStringSplicer.h>
#include <c10/util/hash.h>

#include <set>
#include <sstream>

// Generated template strings (build step: tools/embed_templates.py).
// Not checked in — regenerate with:
//   python tools/embed_templates.py aten/src/ATen/pyre/dispatch/templates/*.mlir \
//       -o aten/src/ATen/pyre/dispatch/PyreKernelTemplates.inc
#include <ATen/pyre/dispatch/PyreKernelTemplates.inc>

namespace at::pyre {

std::string contentHashCacheKey(
    const char* template_sha1,
    const SubstPairs& substitutions,
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

// Build DimSpec vector from the broadcast-aware rule: size==1 is static, else dynamic.
static c10::SmallVector<DimSpec, 6> broadcastAwareDimSpec(
    c10::ArrayRef<int64_t> sizes) {
  c10::SmallVector<DimSpec, 6> spec;
  for (int64_t s : sizes)
    spec.push_back(s == 1 ? DimSpec::fixed(1) : DimSpec::dynamic());
  return spec;
}

// Build DimSpec vector where ALL dims are concrete (fixed).
static c10::SmallVector<DimSpec, 6> concreteDimSpec(
    c10::ArrayRef<int64_t> sizes) {
  c10::SmallVector<DimSpec, 6> spec;
  for (int64_t s : sizes) spec.push_back(DimSpec::fixed(s));
  return spec;
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
// Emit torch.aten.permute MLIR: builds a prim.ListConstruct for the perm,
// then calls torch.aten.permute.
std::string emitPermuteLines(
    const std::string& dst_name,
    const std::string& src_name,
    c10::ArrayRef<int64_t> perm,
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

c10::SmallVector<int64_t, 6> inversePerm(c10::ArrayRef<int64_t> perm) {
  c10::SmallVector<int64_t, 6> inv(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    inv[perm[i]] = static_cast<int64_t>(i);
  return inv;
}

// Compute logical shape from physical shape and adapter permutation.
// The adapter's perm is the stride-sorted order. The inverse perm
// applied to the physical shape gives the logical shape.
static c10::SmallVector<int64_t, 6> logicalShape(
    c10::ArrayRef<int64_t> physical_shape,
    c10::ArrayRef<int64_t> perm) {
  auto inv = inversePerm(perm);
  c10::SmallVector<int64_t, 6> logical(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    logical[i] = physical_shape[inv[i]];
  return logical;
}

static bool isIntegralScalar(double value) {
  return value == static_cast<int64_t>(value);
}

static std::string scalarDecl(const char* name, double value) {
  if (isIntegralScalar(value))
    return std::string("    %") + name + " = torch.constant.int " +
           std::to_string(static_cast<int64_t>(value));
  std::ostringstream ss;
  ss << "    %" << name << " = torch.constant.float " << std::fixed << value;
  return ss.str();
}

static std::string scalarType(double value) {
  return isIntegralScalar(value) ? "!torch.int" : "!torch.float";
}

// ---------------------------------------------------------------------------
// Helpers to resolve torch_op / extra args from a linalg_op name.
// Used by both binary build and generate paths to stay in sync.
// ---------------------------------------------------------------------------

struct BinaryOpInfo {
  std::string torch_op;
  std::string extra_args;
  std::string extra_arg_decls;
  std::string extra_arg_types;
};

static BinaryOpInfo resolveBinaryOp(const std::string& linalg_op) {
  BinaryOpInfo info;
  if (linalg_op == "add") {
    info.torch_op = "torch.aten.add.Tensor";
    info.extra_arg_decls = "    %alpha = torch.constant.int 1";
    info.extra_args = ", %alpha";
    info.extra_arg_types = ", !torch.int";
  } else if (linalg_op == "sub") {
    info.torch_op = "torch.aten.sub.Tensor";
    info.extra_arg_decls = "    %alpha = torch.constant.int 1";
    info.extra_args = ", %alpha";
    info.extra_arg_types = ", !torch.int";
  } else if (linalg_op == "mul") {
    info.torch_op = "torch.aten.mul.Tensor";
  } else if (linalg_op == "div") {
    info.torch_op = "torch.aten.div.Tensor";
  } else if (linalg_op == "mm") {
    info.torch_op = "torch.aten.mm";
  } else if (linalg_op == "pow") {
    info.torch_op = "torch.aten.pow.Tensor_Tensor";
  } else if (linalg_op == "maximum") {
    info.torch_op = "torch.aten.maximum";
  } else if (linalg_op == "minimum") {
    info.torch_op = "torch.aten.minimum";
  } else if (linalg_op == "remainder") {
    info.torch_op = "torch.aten.remainder.Tensor";
  } else if (linalg_op == "fmod") {
    info.torch_op = "torch.aten.fmod.Tensor";
  } else if (linalg_op == "bitwise_and") {
    info.torch_op = "torch.aten.bitwise_and.Tensor";
  } else if (linalg_op == "bitwise_or") {
    info.torch_op = "torch.aten.bitwise_or.Tensor";
  } else if (linalg_op == "bitwise_xor") {
    info.torch_op = "torch.aten.bitwise_xor.Tensor";
  } else if (linalg_op == "atan2") {
    info.torch_op = "torch.aten.atan2";
  } else {
    TORCH_CHECK(false, "pyre: unknown binary op: ", linalg_op);
  }
  return info;
}

// ---------------------------------------------------------------------------
// Helpers to resolve alpha op info. Used by both alpha build and generate.
// ---------------------------------------------------------------------------

struct BinaryAlphaOpInfo {
  std::string torch_op;
  std::string alpha_decl;
};

static BinaryAlphaOpInfo resolveAlphaOp(
    const std::string& alpha_add_op, double alpha_value) {
  BinaryAlphaOpInfo info;
  info.torch_op = (alpha_add_op.find("sub") != std::string::npos)
      ? "torch.aten.sub.Tensor" : "torch.aten.add.Tensor";
  std::ostringstream decl;
  if (alpha_value == static_cast<int64_t>(alpha_value))
    decl << "    %alpha = torch.constant.int "
         << static_cast<int64_t>(alpha_value);
  else
    decl << "    %alpha = torch.constant.float "
         << std::fixed << alpha_value;
  info.alpha_decl = decl.str();
  return info;
}

// ---------------------------------------------------------------------------
// Arg adapter substitution: shared by binary, unary, and mm templates.
//
// For each arg, produces 4 substitution values:
//   {prefix}_input_shape  — physical shape (what the buffer view has)
//   {prefix}_compute_shape — logical shape (what the op sees)
//   {prefix}_adapter      — permute MLIR lines (empty for identity)
//   {prefix}_name         — SSA name to use in the compute op
// ---------------------------------------------------------------------------

struct ArgAdapterVars {
  std::string input_shape;
  std::string compute_shape;
  std::string adapter_body;
  std::string use_name;
};

static ArgAdapterVars buildArgAdapterVars(
    const char* arg_name,
    c10::ArrayRef<int64_t> phys_shape,
    const ArgAdapter& adapter,
    c10::ScalarType dtype) {
  ArgAdapterVars v;
  if (adapter.kind == ArgAdapter::kPermute) {
    auto log_shape = logicalShape(phys_shape, adapter.permutation);
    v.input_shape = concreteShapeStr(phys_shape);
    v.compute_shape = concreteShapeStr(log_shape);
    std::string elt = scalarTypeToTorchMlir(dtype);
    std::string in_type = "!torch.vtensor<[" + v.input_shape + "], " + elt + ">";
    std::string out_type = "!torch.vtensor<[" + v.compute_shape + "], " + elt + ">";
    v.adapter_body = emitPermuteLines(
        arg_name, std::string(arg_name) + "_raw",
        inversePerm(adapter.permutation), in_type, out_type) + "\n";
    v.use_name = arg_name;
  } else {
    v.input_shape = broadcastAwareShapeStr(phys_shape);
    v.compute_shape = v.input_shape;
    v.adapter_body = "";
    v.use_name = std::string(arg_name) + "_raw";
  }
  return v;
}

// ---------------------------------------------------------------------------
// ComputeBody helpers
// ---------------------------------------------------------------------------

static std::string vtensorType(
    const std::string& shape_str, const std::string& elt) {
  return "!torch.vtensor<[" + shape_str + "], " + elt + ">";
}

static std::string builtinTensorType(const std::string& shape_str,
                                      const std::string& elt) {
  // Convert torch-mlir element types to builtin: f32→f32, si32→i32, etc.
  std::string builtin_elt = elt;
  if (builtin_elt.substr(0, 2) == "si")
    builtin_elt = "i" + builtin_elt.substr(2);
  else if (builtin_elt.substr(0, 2) == "ui")
    builtin_elt = "i" + builtin_elt.substr(2);
  else if (builtin_elt == "i1")
    builtin_elt = "i1";
  if (shape_str.empty())
    return "tensor<" + builtin_elt + ">";
  // Convert comma-separated shape to x-separated (tensor<?x4xf32>).
  std::string tensor_dims;
  for (char c : shape_str) {
    if (c == ',') tensor_dims += 'x';
    else tensor_dims += c;
  }
  return "tensor<" + tensor_dims + "x" + builtin_elt + ">";
}

// Wrap a ComputeBody back into a complete MLIR module (backward compat).
std::string wrapComputeBody(
    const std::string& func_name,
    const ComputeBody& body,
    c10::ArrayRef<ArgAdapter> adapters,
    c10::ArrayRef<std::string> physical_input_vtensor_types,
    c10::ArrayRef<std::string> physical_input_names) {
  std::string mlir;

  // Type aliases.
  std::string out_mutable = body.output_vtensor_type;
  // Convert vtensor to mutable tensor for %out_
  // !torch.vtensor<[...], dtype> → !torch.tensor<[...], dtype>
  std::string out_t = out_mutable;
  auto pos = out_t.find("vtensor");
  if (pos != std::string::npos) out_t.replace(pos, 7, "tensor");

  mlir += "!out_t = " + out_t + "\n";
  mlir += "!out_v = " + body.output_vtensor_type + "\n";

  for (size_t i = 0; i < body.input_vtensor_types.size(); ++i) {
    std::string prefix = body.input_names[i];
    // Use the physical type for the input arg (pre-adapter).
    if (i < physical_input_vtensor_types.size()) {
      mlir += "!" + prefix + "_in = " + physical_input_vtensor_types[i] + "\n";
    } else {
      mlir += "!" + prefix + "_in = " + body.input_vtensor_types[i] + "\n";
    }
    mlir += "!" + prefix + "_c = " + body.input_vtensor_types[i] + "\n";
  }

  mlir += "\nmodule @module {\n";
  mlir += "  func.func @" + func_name + "(%out_: !out_t";

  // Input args.
  for (size_t i = 0; i < body.input_names.size(); ++i) {
    std::string phys_name = (i < physical_input_names.size())
        ? physical_input_names[i]
        : body.input_names[i] + "_raw";
    mlir += ", %" + phys_name + ": !" + body.input_names[i] + "_in";
  }

  mlir += ")\n      attributes {torch.assume_strict_symbolic_shapes} {\n";

  // Adapter lines (permute ops for transposed inputs).
  for (size_t i = 0; i < adapters.size() && i < body.input_names.size(); ++i) {
    if (adapters[i].kind == ArgAdapter::kPermute) {
      std::string phys_name = (i < physical_input_names.size())
          ? physical_input_names[i]
          : body.input_names[i] + "_raw";
      std::string in_type = "!" + body.input_names[i] + "_in";
      std::string out_type = "!" + body.input_names[i] + "_c";
      mlir += emitPermuteLines(
          body.input_names[i], phys_name,
          inversePerm(adapters[i].permutation), in_type, out_type);
      mlir += "\n";
    }
  }

  // The torch ops body.
  mlir += body.mlir_ops;

  // Overwrite epilogue.
  mlir += "    torch.overwrite.tensor.contents %result overwrites %out_"
          " : !out_v, !out_t\n";
  mlir += "    return\n";
  mlir += "  }\n";
  mlir += "}\n";

  return mlir;
}

// ---------------------------------------------------------------------------
// ComputeBody generators
// ---------------------------------------------------------------------------

ComputeBody generateBinaryComputeBody(
    const std::string& linalg_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape) {
  auto info = resolveBinaryOp(linalg_op);
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string lhs_s = broadcastAwareShapeStr(lhs_shape);
  std::string rhs_s = broadcastAwareShapeStr(rhs_shape);
  std::string out_s = broadcastAwareShapeStr(out_shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(lhs_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(lhs_shape));
  body.input_vtensor_types.push_back(vtensorType(rhs_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(rhs_shape));
  body.input_tensor_types.push_back(builtinTensorType(lhs_s, elt));
  body.input_tensor_types.push_back(builtinTensorType(rhs_s, elt));
  body.input_names = {"lhs", "rhs"};
  body.output_vtensor_type = vtensorType(out_s, elt);
  body.output_tensor_type = builtinTensorType(out_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  std::string ops;
  if (!info.extra_arg_decls.empty())
    ops += info.extra_arg_decls + "\n";
  ops += "    %result = " + info.torch_op + " %lhs, %rhs" +
         info.extra_args + " : " +
         body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] +
         info.extra_arg_types + " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateBinaryAlphaComputeBody(
    const std::string& alpha_add_op, double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape) {
  auto info = resolveAlphaOp(alpha_add_op, alpha_value);
  std::string alpha_type = scalarType(alpha_value);
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string lhs_s = broadcastAwareShapeStr(lhs_shape);
  std::string rhs_s = broadcastAwareShapeStr(rhs_shape);
  std::string out_s = broadcastAwareShapeStr(out_shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(lhs_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(lhs_shape));
  body.input_vtensor_types.push_back(vtensorType(rhs_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(rhs_shape));
  body.input_tensor_types.push_back(builtinTensorType(lhs_s, elt));
  body.input_tensor_types.push_back(builtinTensorType(rhs_s, elt));
  body.input_names = {"lhs", "rhs"};
  body.output_vtensor_type = vtensorType(out_s, elt);
  body.output_tensor_type = builtinTensorType(out_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  std::string ops;
  ops += info.alpha_decl + "\n";
  ops += "    %result = " + info.torch_op + " %lhs, %rhs, %alpha : " +
         body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] +
         ", " + alpha_type + " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateUnaryComputeBody(
    const std::string& scalar_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string out_s = broadcastAwareShapeStr(out_shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(in_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(input_shape));
  body.input_tensor_types.push_back(builtinTensorType(in_s, elt));
  body.input_names = {"input"};
  body.output_vtensor_type = vtensorType(out_s, elt);
  body.output_tensor_type = builtinTensorType(out_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  std::string ops;
  if (!extra_arg_decls.empty())
    ops += "    " + extra_arg_decls + "\n";
  ops += "    %result = " + scalar_op + " %input" +
         extra_args + " : " +
         body.input_vtensor_types[0] + extra_arg_types +
         " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateMmComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string m1_s = broadcastAwareShapeStr(mat1_shape);
  std::string m2_s = broadcastAwareShapeStr(mat2_shape);
  std::string out_s = broadcastAwareShapeStr(out_shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(m1_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(mat1_shape));
  body.input_vtensor_types.push_back(vtensorType(m2_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(mat2_shape));
  body.input_tensor_types.push_back(builtinTensorType(m1_s, elt));
  body.input_tensor_types.push_back(builtinTensorType(m2_s, elt));
  body.input_names = {"mat1", "mat2"};
  body.output_vtensor_type = vtensorType(out_s, elt);
  body.output_tensor_type = builtinTensorType(out_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  std::string ops;
  ops += "    %result = torch.aten.mm %mat1, %mat2 : " +
         body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] +
         " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateSoftmaxComputeBody(
    const std::string& softmax_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> shape, int64_t dim) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string shape_s = broadcastAwareShapeStr(shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(shape_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(shape));
  body.input_tensor_types.push_back(builtinTensorType(shape_s, elt));
  body.input_names = {"input"};
  body.output_vtensor_type = vtensorType(shape_s, elt);
  body.output_tensor_type = builtinTensorType(shape_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(shape);

  std::string ops;
  ops += "    %dim = torch.constant.int " + std::to_string(dim) + "\n";
  ops += "    %half_to_float = torch.constant.bool false\n";
  ops += "    %result = torch.aten." + softmax_op +
         " %input, %dim, %half_to_float : " +
         body.input_vtensor_types[0] +
         ", !torch.int, !torch.bool -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateComparisonComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string lhs_s = broadcastAwareShapeStr(lhs_shape);
  std::string rhs_s = broadcastAwareShapeStr(rhs_shape);
  std::string out_s = broadcastAwareShapeStr(out_shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(lhs_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(lhs_shape));
  body.input_vtensor_types.push_back(vtensorType(rhs_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(rhs_shape));
  body.input_tensor_types.push_back(builtinTensorType(lhs_s, elt));
  body.input_tensor_types.push_back(builtinTensorType(rhs_s, elt));
  body.input_names = {"lhs", "rhs"};
  body.output_vtensor_type = vtensorType(out_s, "i1");
  body.output_tensor_type = builtinTensorType(out_s, "i1");
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  body.mlir_ops = "    %result = " + torch_op + " %lhs, %rhs : " +
      body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] +
      " -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateComparisonScalarComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, double scalar_value) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);

  ComputeBody body;
  body.input_vtensor_types.push_back(vtensorType(in_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(input_shape));
  body.input_tensor_types.push_back(builtinTensorType(in_s, elt));
  body.input_names = {"input"};
  body.output_vtensor_type = vtensorType(in_s, "i1");
  body.output_tensor_type = builtinTensorType(in_s, "i1");
  body.output_shape_spec = broadcastAwareDimSpec(input_shape);

  std::string ops;
  if (scalar_value == static_cast<int64_t>(scalar_value))
    ops += "    %scalar = torch.constant.int " +
           std::to_string(static_cast<int64_t>(scalar_value)) + "\n";
  else
    ops += "    %scalar = torch.constant.float " +
           std::to_string(scalar_value) + "\n";
  std::string scalar_type = (scalar_value == static_cast<int64_t>(scalar_value))
      ? "!torch.int" : "!torch.float";
  ops += "    %result = " + torch_op + " %input, %scalar : " +
      body.input_vtensor_types[0] + ", " + scalar_type +
      " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

// Forward declaration (defined in the reduction section below).
static std::string reducedShapeStr(
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> dims,
    bool keepdim);

ComputeBody generateBmmComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string m1 = broadcastAwareShapeStr(mat1_shape);
  std::string m2 = broadcastAwareShapeStr(mat2_shape);
  std::string os = broadcastAwareShapeStr(out_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(m1, elt), vtensorType(m2, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(mat1_shape), broadcastAwareDimSpec(mat2_shape)};
  body.input_tensor_types = {builtinTensorType(m1, elt), builtinTensorType(m2, elt)};
  body.input_names = {"mat1", "mat2"};
  body.output_vtensor_type = vtensorType(os, elt);
  body.output_tensor_type = builtinTensorType(os, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);
  body.mlir_ops = "    %result = torch.aten.bmm %mat1, %mat2 : " +
      body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] +
      " -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateWhereComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> cond_shape, c10::ArrayRef<int64_t> self_shape,
    c10::ArrayRef<int64_t> other_shape, c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string cs = broadcastAwareShapeStr(cond_shape);
  std::string ss = broadcastAwareShapeStr(self_shape);
  std::string os_str = broadcastAwareShapeStr(other_shape);
  std::string outs = broadcastAwareShapeStr(out_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(cs, "i1"), vtensorType(ss, elt), vtensorType(os_str, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(cond_shape), broadcastAwareDimSpec(self_shape), broadcastAwareDimSpec(other_shape)};
  body.input_tensor_types = {builtinTensorType(cs, "i1"), builtinTensorType(ss, elt), builtinTensorType(os_str, elt)};
  body.input_names = {"condition", "self_v", "other_v"};
  body.output_vtensor_type = vtensorType(outs, elt);
  body.output_tensor_type = builtinTensorType(outs, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);
  body.mlir_ops = "    %result = torch.aten.where.self %condition, %self_v, %other_v : " +
      body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] + ", " +
      body.input_vtensor_types[2] + " -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateAddmmComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape, c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape, c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string bs = broadcastAwareShapeStr(bias_shape);
  std::string m1 = broadcastAwareShapeStr(mat1_shape);
  std::string m2 = broadcastAwareShapeStr(mat2_shape);
  std::string os = broadcastAwareShapeStr(out_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(bs, elt), vtensorType(m1, elt), vtensorType(m2, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(bias_shape), broadcastAwareDimSpec(mat1_shape), broadcastAwareDimSpec(mat2_shape)};
  body.input_tensor_types = {builtinTensorType(bs, elt), builtinTensorType(m1, elt), builtinTensorType(m2, elt)};
  body.input_names = {"bias", "mat1", "mat2"};
  body.output_vtensor_type = vtensorType(os, elt);
  body.output_tensor_type = builtinTensorType(os, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);
  std::string ops;
  ops += scalarDecl("beta", beta) + "\n";
  ops += scalarDecl("alpha", alpha) + "\n";
  ops += "    %result = torch.aten.addmm %bias, %mat1, %mat2, %beta, %alpha : " +
      body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] + ", " +
      body.input_vtensor_types[2] + ", " + scalarType(beta) + ", " + scalarType(alpha) +
      " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

// Build DimSpec for a reduction output, mirroring reducedShapeStr logic.
// Non-reduced dims are always dynamic (?). Reduced dims become static 1
// (if keepdim) or are removed (if !keepdim).
static c10::SmallVector<DimSpec, 6> reducedDimSpec(
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> dims,
    bool keepdim) {
  std::set<int64_t> reduce_set;
  for (int64_t d : dims) {
    if (d < 0) d += static_cast<int64_t>(input_shape.size());
    reduce_set.insert(d);
  }
  c10::SmallVector<DimSpec, 6> spec;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (reduce_set.count(static_cast<int64_t>(i))) {
      if (keepdim) spec.push_back(DimSpec::fixed(1));
    } else {
      spec.push_back(DimSpec::dynamic());
    }
  }
  return spec;
}

static std::string dimDeclsStr(c10::ArrayRef<int64_t> dims) {
  std::string s;
  for (size_t i = 0; i < dims.size(); ++i)
    s += "    %d" + std::to_string(i) + " = torch.constant.int " +
         std::to_string(dims[i]) + "\n";
  return s;
}
static std::string dimArgsStr(size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) { if (i > 0) s += ", "; s += "%d" + std::to_string(i); }
  return s;
}
static std::string dimTypesStr(size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) { if (i > 0) s += ", "; s += "!torch.int"; }
  return s;
}

ComputeBody generateReductionComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<int64_t> dims, bool keepdim,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string out_s = reducedShapeStr(input_shape, dims, keepdim);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(in_s, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(input_shape)};
  body.input_tensor_types = {builtinTensorType(in_s, elt)};
  body.input_names = {"input"};
  body.output_vtensor_type = vtensorType(out_s, elt);
  body.output_tensor_type = builtinTensorType(out_s, elt);
  body.output_shape_spec = reducedDimSpec(input_shape, dims, keepdim);
  std::string ops;
  ops += dimDeclsStr(dims);
  ops += "    %dims = torch.prim.ListConstruct " + dimArgsStr(dims.size()) +
         " : (" + dimTypesStr(dims.size()) + ") -> !torch.list<int>\n";
  ops += "    %keepdim = torch.constant.bool " + std::string(keepdim ? "true" : "false") + "\n";
  if (!extra_arg_decls.empty()) ops += "    " + extra_arg_decls + "\n";
  ops += "    %result = " + torch_op + " %input, %dims, %keepdim" + extra_args +
         " : " + body.input_vtensor_types[0] + ", !torch.list<int>, !torch.bool" +
         extra_arg_types + " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateSingleDimReductionComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    int64_t dim, bool keepdim,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string out_s = reducedShapeStr(input_shape, {dim}, keepdim);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(in_s, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(input_shape)};
  body.input_tensor_types = {builtinTensorType(in_s, elt)};
  body.input_names = {"input"};
  body.output_vtensor_type = vtensorType(out_s, elt);
  body.output_tensor_type = builtinTensorType(out_s, elt);
  body.output_shape_spec = reducedDimSpec(input_shape, {dim}, keepdim);
  std::string ops;
  ops += "    %dim = torch.constant.int " + std::to_string(dim) + "\n";
  ops += "    %keepdim = torch.constant.bool " + std::string(keepdim ? "true" : "false") + "\n";
  if (!extra_arg_decls.empty()) ops += "    " + extra_arg_decls + "\n";
  ops += "    %result = " + torch_op + " %input, %dim, %keepdim" + extra_args +
         " : " + body.input_vtensor_types[0] + ", !torch.int, !torch.bool" +
         extra_arg_types + " -> " + body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

ComputeBody generateEmbeddingComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string ws = broadcastAwareShapeStr(weight_shape);
  std::string is = broadcastAwareShapeStr(indices_shape);
  std::string os = broadcastAwareShapeStr(out_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(ws, elt), vtensorType(is, "si64")};
  body.input_shape_specs = {broadcastAwareDimSpec(weight_shape), broadcastAwareDimSpec(indices_shape)};
  body.input_tensor_types = {builtinTensorType(ws, elt), builtinTensorType(is, "si64")};
  body.input_names = {"weight", "indices"};
  body.output_vtensor_type = vtensorType(os, elt);
  body.output_tensor_type = builtinTensorType(os, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);
  body.mlir_ops =
      "    %padding_idx = torch.constant.int -1\n"
      "    %scale = torch.constant.bool false\n"
      "    %sparse = torch.constant.bool false\n"
      "    %result = torch.aten.embedding %weight, %indices, %padding_idx, %scale, %sparse : " +
      body.input_vtensor_types[0] + ", " + body.input_vtensor_types[1] +
      ", !torch.int, !torch.bool, !torch.bool -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateIndexSelectComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string ix_s = broadcastAwareShapeStr(index_shape);
  std::string os = broadcastAwareShapeStr(out_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(in_s, elt), vtensorType(ix_s, "si64")};
  body.input_shape_specs = {broadcastAwareDimSpec(input_shape), broadcastAwareDimSpec(index_shape)};
  body.input_tensor_types = {builtinTensorType(in_s, elt), builtinTensorType(ix_s, "si64")};
  body.input_names = {"input", "index"};
  body.output_vtensor_type = vtensorType(os, elt);
  body.output_tensor_type = builtinTensorType(os, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);
  body.mlir_ops = "    %dim = torch.constant.int " + std::to_string(dim) + "\n" +
      "    %result = torch.aten.index_select %input, %dim, %index : " +
      body.input_vtensor_types[0] + ", !torch.int, " + body.input_vtensor_types[1] +
      " -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateGatherComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string ix_s = broadcastAwareShapeStr(index_shape);
  std::string os = broadcastAwareShapeStr(out_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(in_s, elt), vtensorType(ix_s, "si64")};
  body.input_shape_specs = {broadcastAwareDimSpec(input_shape), broadcastAwareDimSpec(index_shape)};
  body.input_tensor_types = {builtinTensorType(in_s, elt), builtinTensorType(ix_s, "si64")};
  body.input_names = {"input", "index"};
  body.output_vtensor_type = vtensorType(os, elt);
  body.output_tensor_type = builtinTensorType(os, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);
  body.mlir_ops = "    %dim = torch.constant.int " + std::to_string(dim) + "\n" +
      "    %sparse_grad = torch.constant.bool false\n" +
      "    %result = torch.aten.gather %input, %dim, %index, %sparse_grad : " +
      body.input_vtensor_types[0] + ", !torch.int, " + body.input_vtensor_types[1] +
      ", !torch.bool -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateScatterSrcComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string ix_s = broadcastAwareShapeStr(index_shape);
  std::string sr_s = broadcastAwareShapeStr(src_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(in_s, elt), vtensorType(ix_s, "si64"), vtensorType(sr_s, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(input_shape), broadcastAwareDimSpec(index_shape), broadcastAwareDimSpec(src_shape)};
  body.input_tensor_types = {builtinTensorType(in_s, elt), builtinTensorType(ix_s, "si64"), builtinTensorType(sr_s, elt)};
  body.input_names = {"self_v", "index", "src"};
  body.output_vtensor_type = vtensorType(in_s, elt);
  body.output_tensor_type = builtinTensorType(in_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(input_shape);
  body.mlir_ops = "    %dim = torch.constant.int " + std::to_string(dim) + "\n" +
      "    %result = torch.aten.scatter.src %self_v, %dim, %index, %src : " +
      body.input_vtensor_types[0] + ", !torch.int, " + body.input_vtensor_types[1] +
      ", " + body.input_vtensor_types[2] + " -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateScatterAddComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_s = broadcastAwareShapeStr(input_shape);
  std::string ix_s = broadcastAwareShapeStr(index_shape);
  std::string sr_s = broadcastAwareShapeStr(src_shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(in_s, elt), vtensorType(ix_s, "si64"), vtensorType(sr_s, elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(input_shape), broadcastAwareDimSpec(index_shape), broadcastAwareDimSpec(src_shape)};
  body.input_tensor_types = {builtinTensorType(in_s, elt), builtinTensorType(ix_s, "si64"), builtinTensorType(sr_s, elt)};
  body.input_names = {"self_v", "index", "src"};
  body.output_vtensor_type = vtensorType(in_s, elt);
  body.output_tensor_type = builtinTensorType(in_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(input_shape);
  body.mlir_ops = "    %dim = torch.constant.int " + std::to_string(dim) + "\n" +
      "    %result = torch.aten.scatter_add %self_v, %dim, %index, %src : " +
      body.input_vtensor_types[0] + ", !torch.int, " + body.input_vtensor_types[1] +
      ", " + body.input_vtensor_types[2] + " -> " + body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateTypeCastComputeBody(
    c10::ScalarType in_dtype, c10::ScalarType out_dtype,
    c10::ArrayRef<int64_t> shape) {
  std::string in_elt = scalarTypeToTorchMlir(in_dtype);
  std::string out_elt = scalarTypeToTorchMlir(out_dtype);
  std::string shape_s = broadcastAwareShapeStr(shape);
  ComputeBody body;
  body.input_vtensor_types = {vtensorType(shape_s, in_elt)};
  body.input_shape_specs = {broadcastAwareDimSpec(shape)};
  body.input_tensor_types = {builtinTensorType(shape_s, in_elt)};
  body.input_names = {"input"};
  body.output_vtensor_type = vtensorType(shape_s, out_elt);
  body.output_tensor_type = builtinTensorType(shape_s, out_elt);
  body.output_shape_spec = broadcastAwareDimSpec(shape);
  std::string dtype_const = "    %dtype = torch.constant.int " +
      std::to_string(static_cast<int64_t>(out_dtype));
  body.mlir_ops = dtype_const + "\n" +
      "    %none = torch.constant.none\n"
      "    %false = torch.constant.bool false\n"
      "    %result = torch.aten._to_copy %input, %dtype, %none, %none, %none, %false, %none : " +
      body.input_vtensor_types[0] +
      ", !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> " +
      body.output_vtensor_type + "\n";
  return body;
}

ComputeBody generateArangeComputeBody(
    c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string size_s = std::to_string(out_size);
  ComputeBody body;
  // Arange has NO tensor inputs.
  body.output_vtensor_type = vtensorType(size_s, elt);
  body.output_tensor_type = builtinTensorType(size_s, elt);
  body.output_shape_spec = concreteDimSpec({out_size});
  body.mlir_ops =
      scalarDecl("start", start) + "\n" +
      scalarDecl("end", end) + "\n" +
      scalarDecl("step", step) + "\n" +
      "    %none = torch.constant.none\n"
      "    %result = torch.aten.arange.start_step %start, %end, %step, %none, %none, %none, %none : " +
      scalarType(start) + ", " + scalarType(end) + ", " + scalarType(step) +
      ", !torch.none, !torch.none, !torch.none, !torch.none -> " +
      body.output_vtensor_type + "\n";
  return body;
}

// ---------------------------------------------------------------------------
// Binary ops — build + generate (unified: identity + permute via template)
// ---------------------------------------------------------------------------

static SubstPairs binaryVars(
    const std::string& func_name,
    const BinaryOpInfo& info,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  ArgAdapter a0 = adapters.size() > 0 ? adapters[0] : ArgAdapter{ArgAdapter::kIdentity, {}};
  ArgAdapter a1 = adapters.size() > 1 ? adapters[1] : ArgAdapter{ArgAdapter::kIdentity, {}};
  auto lhs = buildArgAdapterVars("lhs", lhs_shape, a0, dtype);
  auto rhs = buildArgAdapterVars("rhs", rhs_shape, a1, dtype);
  return {
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"func_name", func_name},
      {"lhs_input_shape", lhs.input_shape},
      {"lhs_compute_shape", lhs.compute_shape},
      {"lhs_adapter", lhs.adapter_body},
      {"lhs_name", lhs.use_name},
      {"rhs_input_shape", rhs.input_shape},
      {"rhs_compute_shape", rhs.compute_shape},
      {"rhs_adapter", rhs.adapter_body},
      {"rhs_name", rhs.use_name},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"torch_op", info.torch_op},
      {"extra_args", info.extra_args},
      {"extra_arg_decls", info.extra_arg_decls},
      {"extra_arg_types", info.extra_arg_types},
  };
}

KernelSpec buildBinaryKernelSpec(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  auto info = resolveBinaryOp(linalg_op);
  return {kTemplate_elementwise_binary_sha1,
          binaryVars(func_name, info, dtype, lhs_shape, rhs_shape, out_shape, adapters)};
}

std::string generateBinaryMlir(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  auto info = resolveBinaryOp(linalg_op);
  return pyreSpliceRange(kTemplate_elementwise_binary,
      binaryVars(func_name, info, dtype, lhs_shape, rhs_shape, out_shape, adapters));
}

// ---------------------------------------------------------------------------
// Binary with alpha — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildBinaryAlphaKernelSpec(
    const std::string& func_name,
    const std::string& alpha_add_op,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  auto info = resolveAlphaOp(alpha_add_op, alpha_value);
  BinaryOpInfo binfo{info.torch_op, ", %alpha", info.alpha_decl,
                     ", " + scalarType(alpha_value)};
  return {kTemplate_elementwise_binary_sha1,
          binaryVars(func_name, binfo, dtype, lhs_shape, rhs_shape, out_shape, adapters)};
}

std::string generateBinaryAlphaMlir(
    const std::string& func_name,
    const std::string& alpha_add_op,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  auto info = resolveAlphaOp(alpha_add_op, alpha_value);
  BinaryOpInfo binfo{info.torch_op, ", %alpha", info.alpha_decl,
                     ", " + scalarType(alpha_value)};
  return pyreSpliceRange(kTemplate_elementwise_binary,
      binaryVars(func_name, binfo, dtype, lhs_shape, rhs_shape, out_shape, adapters));
}

// ---------------------------------------------------------------------------
// Type cast — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildTypeCastKernelSpec(
    const std::string& func_name,
    c10::ScalarType in_dtype,
    c10::ScalarType out_dtype,
    c10::ArrayRef<int64_t> shape) {
  std::string in_elt = scalarTypeToTorchMlir(in_dtype);
  std::string out_elt = scalarTypeToTorchMlir(out_dtype);
  std::string shape_str;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) shape_str += ",";
    shape_str += "?";
  }
  std::string dtype_const = "    %dtype = torch.constant.int " +
      std::to_string(static_cast<int64_t>(out_dtype));
  SubstPairs vars = {
      {"in_element_type", in_elt}, {"out_element_type", out_elt},
      {"func_name", func_name},
      {"input_shape", shape_str}, {"out_shape", shape_str},
      {"dtype_const", dtype_const},
  };
  return {kTemplate_type_cast_sha1, std::move(vars)};
}

std::string generateTypeCastMlir(
    const std::string& func_name,
    c10::ScalarType in_dtype,
    c10::ScalarType out_dtype,
    c10::ArrayRef<int64_t> shape) {
  std::string in_elt = scalarTypeToTorchMlir(in_dtype);
  std::string out_elt = scalarTypeToTorchMlir(out_dtype);
  std::string shape_str;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) shape_str += ",";
    shape_str += "?";
  }
  std::string dtype_const = "    %dtype = torch.constant.int " +
      std::to_string(static_cast<int64_t>(out_dtype));
  SubstPairs vars = {
      {"in_element_type", in_elt}, {"out_element_type", out_elt},
      {"func_name", func_name},
      {"input_shape", shape_str}, {"out_shape", shape_str},
      {"dtype_const", dtype_const},
  };
  return pyreSpliceRange(kTemplate_type_cast, vars);
}

// ---------------------------------------------------------------------------
// bmm — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildBmmKernelSpec(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_shape", broadcastAwareShapeStr(mat2_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
  return {kTemplate_bmm_sha1, std::move(vars)};
}

std::string generateBmmMlir(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_shape", broadcastAwareShapeStr(mat2_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
  return pyreSpliceRange(kTemplate_bmm, vars);
}

// ---------------------------------------------------------------------------
// where — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildWhereKernelSpec(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> cond_shape,
    c10::ArrayRef<int64_t> self_shape,
    c10::ArrayRef<int64_t> other_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"cond_shape", broadcastAwareShapeStr(cond_shape)},
      {"self_shape", broadcastAwareShapeStr(self_shape)},
      {"other_shape", broadcastAwareShapeStr(other_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
  return {kTemplate_where_sha1, std::move(vars)};
}

std::string generateWhereMlir(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> cond_shape,
    c10::ArrayRef<int64_t> self_shape,
    c10::ArrayRef<int64_t> other_shape,
    c10::ArrayRef<int64_t> out_shape) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"cond_shape", broadcastAwareShapeStr(cond_shape)},
      {"self_shape", broadcastAwareShapeStr(self_shape)},
      {"other_shape", broadcastAwareShapeStr(other_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
  return pyreSpliceRange(kTemplate_where, vars);
}

// ---------------------------------------------------------------------------
// Reduction ops — build + generate
// ---------------------------------------------------------------------------

static std::string dimDecls(c10::ArrayRef<int64_t> dims) {
  std::string s;
  for (size_t i = 0; i < dims.size(); ++i) {
    s += "    %d" + std::to_string(i) + " = torch.constant.int " +
         std::to_string(dims[i]) + "\n";
  }
  return s;
}

static std::string dimArgs(size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) s += ", ";
    s += "%d" + std::to_string(i);
  }
  return s;
}

static std::string dimTypes(size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) s += ", ";
    s += "!torch.int";
  }
  return s;
}

static std::string reducedShapeStr(
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> dims,
    bool keepdim) {
  std::set<int64_t> reduce_set;
  for (int64_t d : dims) {
    if (d < 0) d += static_cast<int64_t>(input_shape.size());
    reduce_set.insert(d);
  }
  std::string s;
  bool first = true;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (reduce_set.count(static_cast<int64_t>(i))) {
      if (keepdim) {
        if (!first) s += ",";
        s += "1";
        first = false;
      }
    } else {
      if (!first) s += ",";
      s += "?";
      first = false;
    }
  }
  return s;
}

KernelSpec buildReductionKernelSpec(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<int64_t> dims,
    bool keepdim,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) in_shape += ",";
    in_shape += "?";
  }
  auto out_str = reducedShapeStr(input_shape, dims, keepdim);

  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", in_shape}, {"out_shape", out_str},
      {"torch_op", torch_op},
      {"dim_decls", dimDecls(dims)},
      {"dim_args", dimArgs(dims.size())},
      {"dim_types", dimTypes(dims.size())},
      {"keepdim", keepdim ? "true" : "false"},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
  return {kTemplate_reduction_sha1, std::move(vars)};
}

std::string generateReductionMlir(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<int64_t> dims,
    bool keepdim,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) in_shape += ",";
    in_shape += "?";
  }
  auto out_str = reducedShapeStr(input_shape, dims, keepdim);

  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", in_shape}, {"out_shape", out_str},
      {"torch_op", torch_op},
      {"dim_decls", dimDecls(dims)},
      {"dim_args", dimArgs(dims.size())},
      {"dim_types", dimTypes(dims.size())},
      {"keepdim", keepdim ? "true" : "false"},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
  return pyreSpliceRange(kTemplate_reduction, vars);
}

// ---------------------------------------------------------------------------
// Single-dim reduction ops — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildSingleDimReductionKernelSpec(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    int64_t dim,
    bool keepdim,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) in_shape += ",";
    in_shape += "?";
  }
  auto out_str = reducedShapeStr(input_shape, {dim}, keepdim);

  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", in_shape}, {"out_shape", out_str},
      {"torch_op", torch_op},
      {"dim", std::to_string(dim)},
      {"keepdim", keepdim ? "true" : "false"},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
  return {kTemplate_reduction_single_dim_sha1, std::move(vars)};
}

std::string generateSingleDimReductionMlir(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    int64_t dim,
    bool keepdim,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string in_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) in_shape += ",";
    in_shape += "?";
  }
  auto out_str = reducedShapeStr(input_shape, {dim}, keepdim);

  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", in_shape}, {"out_shape", out_str},
      {"torch_op", torch_op},
      {"dim", std::to_string(dim)},
      {"keepdim", keepdim ? "true" : "false"},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
  return pyreSpliceRange(kTemplate_reduction_single_dim, vars);
}

// ---------------------------------------------------------------------------
// Comparison ops (tensor-tensor) — build + generate
// ---------------------------------------------------------------------------

static SubstPairs comparisonVars(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  ArgAdapter a0 = adapters.size() > 0 ? adapters[0] : ArgAdapter{ArgAdapter::kIdentity, {}};
  ArgAdapter a1 = adapters.size() > 1 ? adapters[1] : ArgAdapter{ArgAdapter::kIdentity, {}};
  auto lhs = buildArgAdapterVars("lhs", lhs_shape, a0, dtype);
  auto rhs = buildArgAdapterVars("rhs", rhs_shape, a1, dtype);
  return {
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"func_name", func_name},
      {"lhs_input_shape", lhs.input_shape},
      {"lhs_compute_shape", lhs.compute_shape},
      {"lhs_adapter", lhs.adapter_body},
      {"lhs_name", lhs.use_name},
      {"rhs_input_shape", rhs.input_shape},
      {"rhs_compute_shape", rhs.compute_shape},
      {"rhs_adapter", rhs.adapter_body},
      {"rhs_name", rhs.use_name},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"torch_op", torch_op},
  };
}

KernelSpec buildComparisonKernelSpec(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  return {kTemplate_comparison_sha1,
          comparisonVars(func_name, torch_op, dtype, lhs_shape, rhs_shape, out_shape, adapters)};
}

std::string generateComparisonMlir(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  return pyreSpliceRange(kTemplate_comparison,
      comparisonVars(func_name, torch_op, dtype, lhs_shape, rhs_shape, out_shape, adapters));
}

// ---------------------------------------------------------------------------
// Comparison ops (tensor-scalar) — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildComparisonScalarKernelSpec(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) shape += ",";
    shape += "?";
  }
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", shape}, {"out_shape", shape},
      {"torch_op", torch_op},
      {"scalar_decl", scalarDecl("scalar", scalar_value)},
      {"scalar_type", scalarType(scalar_value)},
  };
  return {kTemplate_comparison_scalar_sha1, std::move(vars)};
}

std::string generateComparisonScalarMlir(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) shape += ",";
    shape += "?";
  }
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", shape}, {"out_shape", shape},
      {"torch_op", torch_op},
      {"scalar_decl", scalarDecl("scalar", scalar_value)},
      {"scalar_type", scalarType(scalar_value)},
  };
  return pyreSpliceRange(kTemplate_comparison_scalar, vars);
}

// ---------------------------------------------------------------------------
// Scalar binary ops — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildScalarBinaryKernelSpec(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) shape += ",";
    shape += "?";
  }
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", shape}, {"out_shape", shape},
      {"torch_op", torch_op},
      {"scalar_decl", scalarDecl("scalar", scalar_value)},
      {"scalar_type", scalarType(scalar_value)},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
  return {kTemplate_scalar_binary_sha1, std::move(vars)};
}

std::string generateScalarBinaryMlir(
    const std::string& func_name,
    const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) shape += ",";
    shape += "?";
  }
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"input_shape", shape}, {"out_shape", shape},
      {"torch_op", torch_op},
      {"scalar_decl", scalarDecl("scalar", scalar_value)},
      {"scalar_type", scalarType(scalar_value)},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
  return pyreSpliceRange(kTemplate_scalar_binary, vars);
}

// ---------------------------------------------------------------------------
// Unary ops — build + generate
// ---------------------------------------------------------------------------

static SubstPairs unaryVars(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  auto inp = buildArgAdapterVars("input", input_shape, adapter, dtype);
  return {
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"func_name", func_name},
      {"input_input_shape", inp.input_shape},
      {"input_compute_shape", inp.compute_shape},
      {"input_adapter", inp.adapter_body},
      {"input_name", inp.use_name},
      {"out_shape", inp.compute_shape},
      {"torch_op", scalar_op},
      {"extra_arg_decls", extra_arg_decls},
      {"extra_args", extra_args},
      {"extra_arg_types", extra_arg_types},
  };
}

KernelSpec buildUnaryKernelSpec(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  return {kTemplate_elementwise_unary_sha1,
          unaryVars(func_name, scalar_op, dtype, input_shape, out_shape,
                    adapter, extra_arg_decls, extra_args, extra_arg_types)};
}

std::string generateUnaryMlir(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  return pyreSpliceRange(kTemplate_elementwise_unary,
      unaryVars(func_name, scalar_op, dtype, input_shape, out_shape,
                adapter, extra_arg_decls, extra_args, extra_arg_types));
}

// ---------------------------------------------------------------------------
// mm — build + generate
// ---------------------------------------------------------------------------

static SubstPairs mmVars(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  ArgAdapter a0 = adapters.size() > 0 ? adapters[0] : ArgAdapter{ArgAdapter::kIdentity, {}};
  ArgAdapter a1 = adapters.size() > 1 ? adapters[1] : ArgAdapter{ArgAdapter::kIdentity, {}};
  auto m1 = buildArgAdapterVars("mat1", mat1_shape, a0, dtype);
  auto m2 = buildArgAdapterVars("mat2", mat2_shape, a1, dtype);
  return {
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"func_name", func_name},
      {"mat1_input_shape", m1.input_shape}, {"mat1_compute_shape", m1.compute_shape},
      {"mat2_input_shape", m2.input_shape}, {"mat2_compute_shape", m2.compute_shape},
      {"mat1_adapter", m1.adapter_body}, {"mat2_adapter", m2.adapter_body},
      {"mat1_name", m1.use_name}, {"mat2_name", m2.use_name},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
}

KernelSpec buildMmKernelSpec(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  return {kTemplate_mm_sha1,
          mmVars(func_name, dtype, mat1_shape, mat2_shape, out_shape, adapters)};
}

std::string generateMmMlir(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters) {
  return pyreSpliceRange(kTemplate_mm,
      mmVars(func_name, dtype, mat1_shape, mat2_shape, out_shape, adapters));
}


// ---------------------------------------------------------------------------
// addmm — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildAddmmKernelSpec(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_shape", broadcastAwareShapeStr(mat2_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"beta_decl", scalarDecl("beta", beta)},
      {"alpha_decl", scalarDecl("alpha", alpha)},
      {"beta_type", scalarType(beta)},
      {"alpha_type", scalarType(alpha)},
  };
  return {kTemplate_addmm_sha1, std::move(vars)};
}

std::string generateAddmmMlir(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_shape", broadcastAwareShapeStr(mat2_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"beta_decl", scalarDecl("beta", beta)},
      {"alpha_decl", scalarDecl("alpha", alpha)},
      {"beta_type", scalarType(beta)},
      {"alpha_type", scalarType(alpha)},
  };
  return pyreSpliceRange(kTemplate_addmm, vars);
}

// ---------------------------------------------------------------------------
// addmm transposed — build + generate
// ---------------------------------------------------------------------------

KernelSpec buildAddmmTransposedKernelSpec(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape,
    c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string mat2_t = broadcastAwareShapeStr(
      {mat2_orig_shape[1], mat2_orig_shape[0]});
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_orig_shape", broadcastAwareShapeStr(mat2_orig_shape)},
      {"mat2_t_shape", mat2_t},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"beta_decl", scalarDecl("beta", beta)},
      {"alpha_decl", scalarDecl("alpha", alpha)},
      {"beta_type", scalarType(beta)},
      {"alpha_type", scalarType(alpha)},
  };
  return {kTemplate_addmm_transposed_sha1, std::move(vars)};
}

std::string generateAddmmTransposedMlir(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape,
    c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string mat2_t = broadcastAwareShapeStr(
      {mat2_orig_shape[1], mat2_orig_shape[0]});
  SubstPairs vars = {
      {"element_type", elt}, {"func_name", func_name},
      {"bias_shape", broadcastAwareShapeStr(bias_shape)},
      {"mat1_shape", broadcastAwareShapeStr(mat1_shape)},
      {"mat2_orig_shape", broadcastAwareShapeStr(mat2_orig_shape)},
      {"mat2_t_shape", mat2_t},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"beta_decl", scalarDecl("beta", beta)},
      {"alpha_decl", scalarDecl("alpha", alpha)},
      {"beta_type", scalarType(beta)},
      {"alpha_type", scalarType(alpha)},
  };
  return pyreSpliceRange(kTemplate_addmm_transposed, vars);
}

// ---------------------------------------------------------------------------
// PyreKernelAsmFragments
// ---------------------------------------------------------------------------

PyreKernelAsmFragments::PyreKernelAsmFragments(
    std::initializer_list<std::string_view> fragments)
    : fragments_(fragments.begin(), fragments.end()) {
  // Compute combined SHA1 of all fragment texts.
  std::string all;
  for (const auto& f : fragments_) {
    all.append(f.data(), f.size());
    all += '\0';
  }
  combined_digest_ = c10::sha1(all).str();
}

std::string PyreKernelAsmFragments::digest(
    c10::ArrayRef<std::string> compiler_flags,
    const std::function<void(PyreKernelAsmBuilder&)>& recipe) const {
  PyreKernelAsmBuilder builder(*this, PyreKernelAsmBuilder::Mode::kDigest);
  recipe(builder);
  // Append compiler flags to the hash input.
  for (const auto& flag : compiler_flags) {
    builder.hash_input_ += flag;
    builder.hash_input_ += '\0';
  }
  return builder.finish();
}

std::string PyreKernelAsmFragments::generateMlir(
    const std::function<void(PyreKernelAsmBuilder&)>& recipe) const {
  PyreKernelAsmBuilder builder(*this, PyreKernelAsmBuilder::Mode::kGenerate);
  recipe(builder);
  return builder.finish();
}

// ---------------------------------------------------------------------------
// softmax — build + generate
// ---------------------------------------------------------------------------

static SubstPairs softmaxVars(
    const std::string& func_name, const std::string& softmax_op,
    c10::ScalarType dtype, c10::ArrayRef<int64_t> shape, int64_t dim) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"input_shape", broadcastAwareShapeStr(shape)},
      {"out_shape", broadcastAwareShapeStr(shape)},
      {"dim", std::to_string(dim)},
      {"softmax_op", softmax_op},
  };
}

KernelSpec buildSoftmaxKernelSpec(
    const std::string& func_name, const std::string& softmax_op,
    c10::ScalarType dtype, c10::ArrayRef<int64_t> shape, int64_t dim) {
  return {kTemplate_softmax_sha1,
          softmaxVars(func_name, softmax_op, dtype, shape, dim)};
}

std::string generateSoftmaxMlir(
    const std::string& func_name, const std::string& softmax_op,
    c10::ScalarType dtype, c10::ArrayRef<int64_t> shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_softmax,
      softmaxVars(func_name, softmax_op, dtype, shape, dim));
}

// ---------------------------------------------------------------------------
// scatter_src — build + generate
// ---------------------------------------------------------------------------

static SubstPairs scatterSrcVars(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"input_shape", broadcastAwareShapeStr(input_shape)},
      {"out_shape", broadcastAwareShapeStr(input_shape)},
      {"index_shape", broadcastAwareShapeStr(index_shape)},
      {"src_shape", broadcastAwareShapeStr(src_shape)},
      {"dim", std::to_string(dim)},
  };
}

KernelSpec buildScatterSrcKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {kTemplate_scatter_src_sha1,
          scatterSrcVars(func_name, dtype, input_shape, index_shape, src_shape, dim)};
}

std::string generateScatterSrcMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_scatter_src,
      scatterSrcVars(func_name, dtype, input_shape, index_shape, src_shape, dim));
}

// ---------------------------------------------------------------------------
// scatter_add — build + generate
// ---------------------------------------------------------------------------

static SubstPairs scatterAddVars(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"input_shape", broadcastAwareShapeStr(input_shape)},
      {"out_shape", broadcastAwareShapeStr(input_shape)},
      {"index_shape", broadcastAwareShapeStr(index_shape)},
      {"src_shape", broadcastAwareShapeStr(src_shape)},
      {"dim", std::to_string(dim)},
  };
}

KernelSpec buildScatterAddKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {kTemplate_scatter_add_sha1,
          scatterAddVars(func_name, dtype, input_shape, index_shape, src_shape, dim)};
}

std::string generateScatterAddMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_scatter_add,
      scatterAddVars(func_name, dtype, input_shape, index_shape, src_shape, dim));
}

// ---------------------------------------------------------------------------
// scatter_src_inplace / scatter_add_inplace — build + generate
// ---------------------------------------------------------------------------

static SubstPairs scatterInplaceVars(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"index_shape", broadcastAwareShapeStr(index_shape)},
      {"src_shape", broadcastAwareShapeStr(src_shape)},
      {"dim", std::to_string(dim)},
  };
}

KernelSpec buildScatterSrcInplaceKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {kTemplate_scatter_src_inplace_sha1,
          scatterInplaceVars(func_name, dtype, out_shape, index_shape, src_shape, dim)};
}

std::string generateScatterSrcInplaceMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_scatter_src_inplace,
      scatterInplaceVars(func_name, dtype, out_shape, index_shape, src_shape, dim));
}

KernelSpec buildScatterAddInplaceKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return {kTemplate_scatter_add_inplace_sha1,
          scatterInplaceVars(func_name, dtype, out_shape, index_shape, src_shape, dim)};
}

std::string generateScatterAddInplaceMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_scatter_add_inplace,
      scatterInplaceVars(func_name, dtype, out_shape, index_shape, src_shape, dim));
}

// ---------------------------------------------------------------------------
// embedding — build + generate
// ---------------------------------------------------------------------------

static SubstPairs embeddingVars(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"weight_shape", broadcastAwareShapeStr(weight_shape)},
      {"indices_shape", broadcastAwareShapeStr(indices_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
  };
}

KernelSpec buildEmbeddingKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape) {
  return {kTemplate_embedding_sha1,
          embeddingVars(func_name, dtype, weight_shape, indices_shape, out_shape)};
}

std::string generateEmbeddingMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape) {
  return pyreSpliceRange(kTemplate_embedding,
      embeddingVars(func_name, dtype, weight_shape, indices_shape, out_shape));
}

// ---------------------------------------------------------------------------
// index_select — build + generate
// ---------------------------------------------------------------------------

static SubstPairs indexSelectVars(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"input_shape", broadcastAwareShapeStr(input_shape)},
      {"index_shape", broadcastAwareShapeStr(index_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"dim", std::to_string(dim)},
  };
}

KernelSpec buildIndexSelectKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  return {kTemplate_index_select_sha1,
          indexSelectVars(func_name, dtype, input_shape, index_shape, out_shape, dim)};
}

std::string generateIndexSelectMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_index_select,
      indexSelectVars(func_name, dtype, input_shape, index_shape, out_shape, dim));
}

// ---------------------------------------------------------------------------
// gather — build + generate
// ---------------------------------------------------------------------------

static SubstPairs gatherVars(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  return {
      {"func_name", func_name},
      {"element_type", scalarTypeToTorchMlir(dtype)},
      {"input_shape", broadcastAwareShapeStr(input_shape)},
      {"index_shape", broadcastAwareShapeStr(index_shape)},
      {"out_shape", broadcastAwareShapeStr(out_shape)},
      {"dim", std::to_string(dim)},
  };
}

KernelSpec buildGatherKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  return {kTemplate_gather_sha1,
          gatherVars(func_name, dtype, input_shape, index_shape, out_shape, dim)};
}

std::string generateGatherMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim) {
  return pyreSpliceRange(kTemplate_gather,
      gatherVars(func_name, dtype, input_shape, index_shape, out_shape, dim));
}

// ---------------------------------------------------------------------------
// arange — build + generate
// ---------------------------------------------------------------------------

static SubstPairs arangeVars(
    const std::string& func_name, c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step) {
  std::string elt = scalarTypeToTorchMlir(dtype);
  bool is_float = isFloatDtype(dtype);
  std::string scalar_type = is_float ? "!torch.float" : "!torch.int";
  std::string size_str = std::to_string(out_size);

  std::ostringstream ss;
  ss << std::fixed;
  std::string start_decl, end_decl, step_decl;
  if (is_float) {
    ss.str(""); ss << "    %start = torch.constant.float " << start;
    start_decl = ss.str();
    ss.str(""); ss << "    %end = torch.constant.float " << end;
    end_decl = ss.str();
    ss.str(""); ss << "    %step = torch.constant.float " << step;
    step_decl = ss.str();
  } else {
    start_decl = "    %start = torch.constant.int " +
        std::to_string(static_cast<int64_t>(start));
    end_decl = "    %end = torch.constant.int " +
        std::to_string(static_cast<int64_t>(end));
    step_decl = "    %step = torch.constant.int " +
        std::to_string(static_cast<int64_t>(step));
  }

  return {
      {"func_name", func_name}, {"element_type", elt},
      {"out_size", size_str}, {"scalar_type", scalar_type},
      {"start_decl", start_decl}, {"end_decl", end_decl},
      {"step_decl", step_decl},
  };
}

KernelSpec buildArangeKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step) {
  return {kTemplate_arange_sha1,
          arangeVars(func_name, dtype, out_size, start, end, step)};
}

std::string generateArangeMlir(
    const std::string& func_name, c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step) {
  return pyreSpliceRange(kTemplate_arange,
      arangeVars(func_name, dtype, out_size, start, end, step));
}

} // namespace at::pyre
