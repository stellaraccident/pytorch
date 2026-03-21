// Ternary where template.
// Three tensors (condition: bool, self, other) -> one tensor.
//
// Placeholders: element_type, func_name, cond_shape, self_shape,
//   other_shape, out_shape

!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!cond = !torch.vtensor<[$$cond_shape$$], i1>
!self_v = !torch.vtensor<[$$self_shape$$], $$element_type$$>
!other_v = !torch.vtensor<[$$other_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %condition: !cond, %self: !self_v, %other: !other_v)
      attributes {torch.assume_strict_symbolic_shapes} {
    %result = torch.aten.where.self %condition, %self, %other : !cond, !self_v, !other_v -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
