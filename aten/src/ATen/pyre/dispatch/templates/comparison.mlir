// Comparison binary template (eq.Tensor, lt.Tensor, etc.)
// Two tensors -> one bool tensor (with broadcast).
//
// Placeholders: element_type, func_name, lhs_shape, rhs_shape,
//   out_shape, torch_op

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], i1>,
      %lhs: !torch.vtensor<[$$lhs_shape$$], $$element_type$$>,
      %rhs: !torch.vtensor<[$$rhs_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %result = $$torch_op$$ %lhs, %rhs : !torch.vtensor<[$$lhs_shape$$], $$element_type$$>, !torch.vtensor<[$$rhs_shape$$], $$element_type$$> -> !torch.vtensor<[$$out_shape$$], i1>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], i1>, !torch.tensor<[$$out_shape$$], i1>
    return
  }
}
