// Reduction template (sum, mean, amax, etc.)
// One tensor + dim list + keepdim -> one tensor.
//
// Placeholders: element_type, func_name, input_shape, out_shape,
//   torch_op, dim_decls, dim_args, dim_types, keepdim,
//   extra_arg_decls, extra_args, extra_arg_types

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %input: !torch.vtensor<[$$input_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    $$dim_decls$$
    %dims = torch.prim.ListConstruct $$dim_args$$ : ($$dim_types$$) -> !torch.list<int>
    %keepdim = torch.constant.bool $$keepdim$$
    $$extra_arg_decls$$
    %result = $$torch_op$$ %input, %dims, %keepdim$$extra_args$$ : !torch.vtensor<[$$input_shape$$], $$element_type$$>, !torch.list<int>, !torch.bool$$extra_arg_types$$ -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
