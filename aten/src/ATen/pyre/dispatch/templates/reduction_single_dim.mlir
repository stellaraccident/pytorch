// Single-dim reduction template (prod.dim_int).
// One tensor + single int dim + keepdim -> one tensor.
//
// Placeholders: element_type, func_name, input_shape, out_shape,
//   torch_op, dim, keepdim, extra_arg_decls, extra_args, extra_arg_types

!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!input = !torch.vtensor<[$$input_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %input: !input)
      attributes {torch.assume_strict_symbolic_shapes} {
    %dim = torch.constant.int $$dim$$
    %keepdim = torch.constant.bool $$keepdim$$
    $$extra_arg_decls$$
    %result = $$torch_op$$ %input, %dim, %keepdim$$extra_args$$ : !input, !torch.int, !torch.bool$$extra_arg_types$$ -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
