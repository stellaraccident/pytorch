!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!lhs_in = !torch.vtensor<[$$lhs_input_shape$$], $$element_type$$>
!lhs_c = !torch.vtensor<[$$lhs_compute_shape$$], $$element_type$$>
!rhs_in = !torch.vtensor<[$$rhs_input_shape$$], $$element_type$$>
!rhs_c = !torch.vtensor<[$$rhs_compute_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %lhs_raw: !lhs_in, %rhs_raw: !rhs_in)
      attributes {torch.assume_strict_symbolic_shapes} {
$$lhs_adapter$$$$rhs_adapter$$    $$extra_arg_decls$$
    %result = $$torch_op$$ %$$lhs_name$$, %$$rhs_name$$$$extra_args$$ : !lhs_c, !rhs_c$$extra_arg_types$$ -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
