// Scalar binary template (add.Scalar, mul.Scalar, etc.)
// One tensor + one scalar constant -> one tensor.
//
// Placeholders: element_type, func_name, input_shape, out_shape,
//   torch_op, scalar_decl, scalar_type,
//   extra_arg_decls, extra_args, extra_arg_types

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %input: !torch.vtensor<[$$input_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    $$scalar_decl$$
    $$extra_arg_decls$$
    %result = $$torch_op$$ %input, %scalar$$extra_args$$ : !torch.vtensor<[$$input_shape$$], $$element_type$$>, $$scalar_type$$$$extra_arg_types$$ -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
