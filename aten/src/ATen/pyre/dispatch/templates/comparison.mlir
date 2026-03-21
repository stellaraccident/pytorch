// Comparison binary template (eq.Tensor, lt.Tensor, etc.)
// Two tensors -> one bool tensor (with broadcast).
//
// Placeholders: element_type, func_name, lhs_shape, rhs_shape,
//   out_shape, torch_op

!out_t = !torch.tensor<[$$out_shape$$], i1>
!out_v = !torch.vtensor<[$$out_shape$$], i1>
!lhs = !torch.vtensor<[$$lhs_shape$$], $$element_type$$>
!rhs = !torch.vtensor<[$$rhs_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %lhs: !lhs, %rhs: !rhs)
      attributes {torch.assume_strict_symbolic_shapes} {
    %result = $$torch_op$$ %lhs, %rhs : !lhs, !rhs -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
