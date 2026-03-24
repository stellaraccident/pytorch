#include <ATen/pyre/dispatch/PyreAbiGenerator.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/pyre/impl/PyreStorage.h>

#include <sstream>

namespace at::pyre {

// ---------------------------------------------------------------------------
// Type string helpers
// ---------------------------------------------------------------------------

std::string AbiGenerator::builtinElementType(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float: return "f32";
    case c10::ScalarType::Double: return "f64";
    case c10::ScalarType::Half: return "f16";
    case c10::ScalarType::BFloat16: return "bf16";
    case c10::ScalarType::Int: return "i32";
    case c10::ScalarType::Long: return "i64";
    case c10::ScalarType::Short: return "i16";
    case c10::ScalarType::Byte: return "i8";
    case c10::ScalarType::Char: return "i8";
    case c10::ScalarType::Bool: return "i1";
    default:
      TORCH_CHECK(false, "pyre: unsupported dtype for builtin tensor: ",
                  c10::toString(dtype));
  }
}

std::string AbiGenerator::flatTensorType(c10::ScalarType dtype) {
  return "tensor<?x" + builtinElementType(dtype) + ">";
}

std::string AbiGenerator::shapedTensorType(
    c10::ArrayRef<int64_t> sizes, c10::ScalarType dtype) {
  std::string s = "tensor<";
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 1) s += "1";
    else s += "?";
    s += "x";
  }
  s += builtinElementType(dtype) + ">";
  return s;
}

// ---------------------------------------------------------------------------
// Tensor visit
// ---------------------------------------------------------------------------

void AbiGenerator::visitTensor(const at::Tensor& t, bool is_output) {
  auto* storage_impl = t.storage().unsafeGetStorageImpl();

  int buf_idx = -1;
  for (int i = 0; i < static_cast<int>(unique_bufs_.size()); ++i) {
    if (unique_bufs_[i].storage == storage_impl) {
      buf_idx = i;
      break;
    }
  }
  if (buf_idx < 0) {
    buf_idx = static_cast<int>(unique_bufs_.size());
    // Total elements in the storage allocation.
    int64_t total_bytes = static_cast<int64_t>(
        t.storage().nbytes());
    int64_t elem_sz = t.element_size();
    int64_t total_elems = total_bytes / elem_sz;
    unique_bufs_.push_back({storage_impl, total_elems, t.scalar_type()});
  }

  int64_t element_offset = t.storage_offset();
  int64_t elem_size = t.element_size();
  int byte_alignment = AbiPacker::computeByteAlignment(element_offset, elem_size);

  ArgAdapter adapter = ArgAdapter::analyze(t);

  TensorInfo info;
  info.buf_idx = buf_idx;
  info.element_offset = element_offset;
  info.byte_alignment = byte_alignment;
  info.is_output = is_output;
  info.adapter = std::move(adapter);
  info.sizes = {t.sizes().begin(), t.sizes().end()};
  // Compute physical sizes for permuted tensors.
  if (info.adapter.kind == ArgAdapter::kPermute) {
    info.phys_sizes.resize(info.sizes.size());
    for (size_t d = 0; d < info.sizes.size(); ++d)
      info.phys_sizes[d] = info.sizes[info.adapter.permutation[d]];
  } else {
    info.phys_sizes = info.sizes;
  }
  info.dtype = t.scalar_type();
  info.elem_size = elem_size;
  tensors_.push_back(std::move(info));
}

void AbiGenerator::visitInput(const at::Tensor& t) {
  visitTensor(t, false);
}

void AbiGenerator::visitOutput(const at::Tensor& t) {
  visitTensor(t, true);
}

// ---------------------------------------------------------------------------
// Compute function generation
// ---------------------------------------------------------------------------

std::string AbiGenerator::emitComputeFunction(
    const ComputeBody& body) const {
  std::ostringstream ss;

  // func.func private @compute(...) -> result_type
  // Must be func.func (not util.func) for torch-mlir lowering.
  ss << "  func.func private @compute(";
  for (size_t i = 0; i < body.input_names.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << "%" << body.input_names[i] << "_builtin: "
       << body.input_tensor_types[i];
  }
  ss << ") -> " << body.output_tensor_type << "\n";
  ss << "      attributes {inlining_policy = #util.inline.always} {\n";

  // torch_c.from_builtin_tensor for each input.
  for (size_t i = 0; i < body.input_names.size(); ++i) {
    ss << "    %" << body.input_names[i]
       << " = torch_c.from_builtin_tensor %" << body.input_names[i]
       << "_builtin : " << body.input_tensor_types[i]
       << " -> " << body.input_vtensor_types[i] << "\n";
  }

  // The torch op body.
  ss << body.mlir_ops;

  // torch_c.to_builtin_tensor for result.
  ss << "    %result_builtin = torch_c.to_builtin_tensor %result : "
     << body.output_vtensor_type << " -> " << body.output_tensor_type << "\n";
  ss << "    return %result_builtin : " << body.output_tensor_type << "\n";
  ss << "  }\n";

  return ss.str();
}

// ---------------------------------------------------------------------------
// Envelope function generation
// ---------------------------------------------------------------------------

std::string AbiGenerator::emitEnvelopeFunction(
    const std::string& envelope_name,
    const ComputeBody& body) const {
  std::ostringstream ss;

  // Determine which unique buffers are used by inputs (not output-only).
  c10::SmallVector<bool, 4> buf_used_by_input(unique_bufs_.size(), false);
  for (const auto& t : tensors_) {
    if (!t.is_output)
      buf_used_by_input[t.buf_idx] = true;
  }

  // Collect which tensors have non-zero offsets.
  c10::SmallVector<int, 8> offset_tensor_indices;
  for (int i = 0; i < static_cast<int>(tensors_.size()); ++i) {
    if (tensors_[i].element_offset != 0 && !tensors_[i].is_output)
      offset_tensor_indices.push_back(i);
  }

  // Collect dynamic dim indices from PHYSICAL sizes of input tensors.
  // For permuted tensors, phys_sizes reflects the memory layout.
  c10::SmallVector<std::pair<int, int>, 16> dynamic_dims;  // (tensor_idx, dim_idx)
  for (int i = 0; i < static_cast<int>(tensors_.size()); ++i) {
    if (tensors_[i].is_output) continue;
    for (int d = 0; d < static_cast<int>(tensors_[i].phys_sizes.size()); ++d) {
      if (tensors_[i].phys_sizes[d] != 1)
        dynamic_dims.push_back({i, d});
    }
  }

  // --- Envelope signature ---
  ss << "  util.func public @" << envelope_name << "(\n";

  // Unique input buffers as opaque !hal.buffer + element count.
  for (int i = 0; i < static_cast<int>(unique_bufs_.size()); ++i) {
    if (!buf_used_by_input[i]) continue;
    ss << "      %buf" << i << ": !hal.buffer,\n";
    ss << "      %buf" << i << "_elems: index,\n";
  }

  // Element offsets for non-zero offset tensors.
  for (int i = 0; i < static_cast<int>(offset_tensor_indices.size()); ++i) {
    ss << "      %off_" << offset_tensor_indices[i] << ": index,\n";
  }

  // Dynamic dims.
  for (int i = 0; i < static_cast<int>(dynamic_dims.size()); ++i) {
    ss << "      %dim_" << dynamic_dims[i].first
       << "_" << dynamic_dims[i].second << ": index,\n";
  }

  // Output buffers.
  for (int i = 0; i < static_cast<int>(tensors_.size()); ++i) {
    if (tensors_[i].is_output) {
      ss << "      %buf_out_" << i << ": !hal.buffer,\n";
    }
  }

  // Transients, wait, signal.
  ss << "      %transients: !hal.buffer,\n";
  ss << "      %wait: !hal.fence,\n";
  ss << "      %signal: !hal.fence\n";
  ss << "  ) {\n";

  // --- Import unique INPUT buffers as flat 1D tensors ---
  for (int i = 0; i < static_cast<int>(unique_bufs_.size()); ++i) {
    if (!buf_used_by_input[i]) continue;
    auto flat_type = flatTensorType(unique_bufs_[i].dtype);
    ss << "    %parent_" << i << " = hal.tensor.import wait(%wait) => %buf"
       << i << " \"buf" << i << "\"\n"
       << "        : !hal.buffer -> " << flat_type
       << "{%buf" << i << "_elems}\n";
  }

  // --- Slice operands from parents (for offset tensors) ---
  // Also shape non-offset tensors from flat to shaped.
  int input_idx = 0;
  for (int ti = 0; ti < static_cast<int>(tensors_.size()); ++ti) {
    if (tensors_[ti].is_output) continue;
    const auto& t = tensors_[ti];
    auto flat_type = flatTensorType(t.dtype);

    // Use physical sizes for import — these match what's in memory.
    const auto& import_sizes = t.phys_sizes;

    // Compute element count (product of physical dims).
    std::string count_var;
    if (import_sizes.size() == 1 && import_sizes[0] != 1) {
      count_var = "%dim_" + std::to_string(ti) + "_0";
    } else {
      std::string prev;
      int mul_idx = 0;
      for (size_t d = 0; d < import_sizes.size(); ++d) {
        if (import_sizes[d] == 1) continue;
        std::string dim_val = "%dim_" + std::to_string(ti) + "_" +
                              std::to_string(d);
        if (prev.empty()) {
          prev = dim_val;
        } else {
          std::string result = "%count_" + std::to_string(ti) + "_" +
                               std::to_string(mul_idx++);
          ss << "    " << result << " = arith.muli " << prev << ", "
             << dim_val << " : index\n";
          prev = result;
        }
      }
      if (prev.empty()) {
        ss << "    %c1_count_" << ti << " = arith.constant 1 : index\n";
        count_var = "%c1_count_" + std::to_string(ti);
      } else {
        count_var = prev;
      }
    }

    // Import using physical shape (for permuted tensors, this differs from logical).
    auto phys_shaped_type = shapedTensorType(import_sizes, t.dtype);
    std::string shaped_name = "%shaped_" + std::to_string(ti);

    if (import_sizes.empty()) {
      // 0-dim scalar tensor: import directly as scalar tensor.
      auto scalar_type = "tensor<" + builtinElementType(t.dtype) + ">";
      ss << "    " << shaped_name
         << " = hal.tensor.import wait(%wait) => %buf"
         << t.buf_idx << " \"input" << ti << "\"\n"
         << "        : !hal.buffer -> " << scalar_type << "\n";
    } else if (t.element_offset == 0 && import_sizes.size() == 1) {
      // Simple 1D case: extract_slice from flat parent.
      ss << "    %c0_" << ti << " = arith.constant 0 : index\n";
      ss << "    " << shaped_name << " = tensor.extract_slice %parent_"
         << t.buf_idx << "[%c0_" << ti << "] [" << count_var << "] [1]\n"
         << "        : " << flat_type << " to " << flat_type << "\n";
    } else if (t.element_offset == 0) {
      // Multi-dim, no offset: import directly as physical shaped tensor.
      ss << "    " << shaped_name
         << " = hal.tensor.import wait(%wait) => %buf"
         << t.buf_idx << " \"input" << ti << "\"\n"
         << "        : !hal.buffer -> " << phys_shaped_type << "{";
      bool first_d = true;
      for (size_t d = 0; d < import_sizes.size(); ++d) {
        if (import_sizes[d] != 1) {
          if (!first_d) ss << ", ";
          ss << "%dim_" << ti << "_" << d;
          first_d = false;
        }
      }
      ss << "}\n";
    } else {
      // Non-zero offset: import flat, extract_slice at offset.
      std::string off_var = "%off_" + std::to_string(ti);
      int byte_align = t.byte_alignment;
      int elem_align = byte_align / static_cast<int>(t.elem_size);
      if (elem_align < 1) elem_align = 1;
      if (elem_align > 1) {
        ss << "    %off_aligned_" << ti << " = util.assume.int "
           << off_var << "<udiv = " << elem_align << "> : index\n";
        off_var = "%off_aligned_" + std::to_string(ti);
      }
      std::string flat_name = "%flat_" + std::to_string(ti);
      ss << "    " << flat_name << " = tensor.extract_slice %parent_"
         << t.buf_idx << "[" << off_var << "] [" << count_var << "] [1]\n"
         << "        : " << flat_type << " to " << flat_type << "\n";

      if (import_sizes.size() <= 1) {
        shaped_name = flat_name;
      } else {
        // Reshape flat → shaped via expand_shape.
        // Use fully-dynamic type so all dims accept SSA values.
        std::string dyn_shaped_type = "tensor<";
        for (size_t d = 0; d < import_sizes.size(); ++d) {
          dyn_shaped_type += "?x";
        }
        dyn_shaped_type += builtinElementType(t.dtype) + ">";

        for (size_t d = 0; d < import_sizes.size(); ++d) {
          if (import_sizes[d] == 1) {
            ss << "    %c1_dim_" << ti << "_" << d
               << " = arith.constant 1 : index\n";
          }
        }
        ss << "    " << shaped_name << " = tensor.expand_shape "
           << flat_name << " [[0";
        for (size_t d = 1; d < import_sizes.size(); ++d) ss << ", " << d;
        ss << "]]\n        output_shape [";
        for (size_t d = 0; d < import_sizes.size(); ++d) {
          if (d > 0) ss << ", ";
          if (import_sizes[d] == 1)
            ss << "%c1_dim_" << ti << "_" << d;
          else
            ss << "%dim_" << ti << "_" << d;
        }
        ss << "] : " << flat_type << " into " << dyn_shaped_type << "\n";
      }
    }

    // Apply permutation if needed (physical → logical via linalg.transpose).
    std::string compute_name = "%compute_in_" + std::to_string(ti);
    auto logical_type = body.input_tensor_types[input_idx];
    if (t.adapter.kind == ArgAdapter::kPermute) {
      auto inv = inversePerm(t.adapter.permutation);
      auto empty_name = "%empty_perm_" + std::to_string(ti);
      // Emit tensor.empty with logical dim values.
      // The inverse perm maps logical dim → physical dim.
      // Physical dim d has value %dim_{ti}_{d} (or 1 for static).
      ss << "    " << empty_name << " = tensor.empty(";
      bool first_dim = true;
      for (size_t ld = 0; ld < inv.size(); ++ld) {
        int64_t phys_d = inv[ld];
        if (t.phys_sizes[phys_d] == 1) continue;  // static dim, not in dynamic args
        if (!first_dim) ss << ", ";
        ss << "%dim_" << ti << "_" << phys_d;
        first_dim = false;
      }
      ss << ") : " << logical_type << "\n";
      ss << "    " << compute_name << " = linalg.transpose ins("
         << shaped_name << " : " << phys_shaped_type << ")\n"
         << "        outs(" << empty_name << " : " << logical_type
         << ") permutation = [";
      for (size_t i = 0; i < inv.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << inv[i];
      }
      ss << "]\n";
    } else {
      compute_name = shaped_name;
    }

    input_idx++;
  }

  // --- Call compute function ---
  ss << "\n    %result = func.call @compute(";
  input_idx = 0;
  for (int ti = 0; ti < static_cast<int>(tensors_.size()); ++ti) {
    if (tensors_[ti].is_output) continue;
    if (input_idx > 0) ss << ", ";
    if (tensors_[ti].adapter.kind == ArgAdapter::kPermute)
      ss << "%compute_in_" << ti;
    else
      ss << "%shaped_" << ti;
    input_idx++;
  }
  ss << ")\n        : (";
  for (size_t i = 0; i < body.input_tensor_types.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << body.input_tensor_types[i];
  }
  ss << ") -> " << body.output_tensor_type << "\n";

  // --- Output handling ---
  // Find output tensor info.
  int out_ti = -1;
  for (int ti = 0; ti < static_cast<int>(tensors_.size()); ++ti) {
    if (tensors_[ti].is_output) { out_ti = ti; break; }
  }
  TORCH_CHECK(out_ti >= 0, "pyre: AbiGenerator has no output tensor");

  const auto& out = tensors_[out_ti];

  // Determine which output dims are dynamic (? in the type).
  // Parse body.output_tensor_type to find which dims are dynamic.
  // A dim is dynamic if it appears as '?' in the type string.
  c10::SmallVector<bool, 6> out_dim_is_dynamic;
  {
    // Parse "tensor<?x4xf32>" style type to find dynamic dims.
    auto type_str = body.output_tensor_type;
    auto lt = type_str.find('<');
    auto gt = type_str.rfind('x');
    if (lt != std::string::npos) {
      std::string dims_str = type_str.substr(lt + 1);
      // Remove element type suffix (last xTYPE>)
      for (size_t d = 0; d < out.sizes.size(); ++d) {
        // Find next dim token before 'x' or '>'
        size_t pos = 0;
        std::string remaining = dims_str;
        out_dim_is_dynamic.clear();
        for (size_t dd = 0; dd < out.sizes.size(); ++dd) {
          if (remaining.empty()) break;
          if (remaining[0] == '?') {
            out_dim_is_dynamic.push_back(true);
            remaining = remaining.substr(1);
          } else {
            out_dim_is_dynamic.push_back(false);
            // Skip digits
            size_t end = 0;
            while (end < remaining.size() && std::isdigit(remaining[end])) ++end;
            remaining = remaining.substr(end);
          }
          // Skip 'x' separator
          if (!remaining.empty() && remaining[0] == 'x')
            remaining = remaining.substr(1);
        }
        break; // only one pass needed
      }
    }
  }
  // Fallback: if parsing failed, assume non-1 dims are dynamic.
  if (out_dim_is_dynamic.size() != out.sizes.size()) {
    out_dim_is_dynamic.clear();
    for (size_t d = 0; d < out.sizes.size(); ++d)
      out_dim_is_dynamic.push_back(out.sizes[d] != 1);
  }

  // Emit tensor.dim for dynamic output dims.
  for (size_t d = 0; d < out.sizes.size(); ++d) {
    if (out_dim_is_dynamic[d]) {
      ss << "    %c" << d << "_out_idx = arith.constant " << d << " : index\n";
      ss << "    %dim_out_" << d << " = tensor.dim %result, %c"
         << d << "_out_idx : " << body.output_tensor_type << "\n";
    }
  }

  // Helper lambda to emit dim list in {%d0, %d1} format.
  auto emitDynDims = [&]() {
    bool first = true;
    for (size_t d = 0; d < out.sizes.size(); ++d) {
      if (out_dim_is_dynamic[d]) {
        if (!first) ss << ", ";
        ss << "%dim_out_" << d;
        first = false;
      }
    }
  };

  // Alias result to output buffer.
  ss << "\n    %aliased = hal.tensor.alias wait(%wait) =>\n"
     << "        %result : " << body.output_tensor_type << "{";
  emitDynDims();
  ss << "} to %buf_out_" << out_ti << " : !hal.buffer\n";

  // Transients annotation.
  ss << "    %annotated = hal.tensor.transients %aliased : "
     << body.output_tensor_type << "{";
  emitDynDims();
  ss << "}\n        from %transients : !hal.buffer\n";

  // Barrier → signal.
  ss << "    %ready = hal.tensor.barrier join(%annotated : "
     << body.output_tensor_type << ")\n"
     << "        => %signal : !hal.fence\n";

  // Export (discarded).
  ss << "    %out_bv = hal.tensor.export %ready \"output0\"\n"
     << "        : " << body.output_tensor_type << "{";
  emitDynDims();
  ss << "} -> !hal.buffer_view\n";

  ss << "    util.return\n";
  ss << "  }\n";

  return ss.str();
}

// ---------------------------------------------------------------------------
// Full module generation
// ---------------------------------------------------------------------------

std::string AbiGenerator::generateModule(
    const std::string& envelope_name,
    const ComputeBody& body) const {
  std::ostringstream ss;

  ss << "module @module {\n\n";
  ss << emitComputeFunction(body);
  ss << "\n";
  ss << emitEnvelopeFunction(envelope_name, body);
  ss << "}\n";

  return ss.str();
}

} // namespace at::pyre
