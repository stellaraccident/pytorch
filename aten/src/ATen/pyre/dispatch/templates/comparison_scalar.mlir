// Comparison scalar template (eq.Scalar, lt.Scalar, etc.)
// One tensor + scalar -> one bool tensor.
//
// Placeholders: element_type, func_name, input_shape, out_shape,
//   torch_op, scalar_decl, scalar_type

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], i1>,
      %input: !torch.vtensor<[$$input_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    $$scalar_decl$$
    %result = $$torch_op$$ %input, %scalar : !torch.vtensor<[$$input_shape$$], $$element_type$$>, $$scalar_type$$ -> !torch.vtensor<[$$out_shape$$], i1>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], i1>, !torch.tensor<[$$out_shape$$], i1>
    return
  }
}
