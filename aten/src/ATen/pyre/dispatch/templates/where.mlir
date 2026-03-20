// Ternary where template.
// Three tensors (condition: bool, self, other) -> one tensor.
//
// Placeholders: element_type, func_name, cond_shape, self_shape,
//   other_shape, out_shape

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %condition: !torch.vtensor<[$$cond_shape$$], i1>,
      %self: !torch.vtensor<[$$self_shape$$], $$element_type$$>,
      %other: !torch.vtensor<[$$other_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %result = torch.aten.where.self %condition, %self, %other : !torch.vtensor<[$$cond_shape$$], i1>, !torch.vtensor<[$$self_shape$$], $$element_type$$>, !torch.vtensor<[$$other_shape$$], $$element_type$$> -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
