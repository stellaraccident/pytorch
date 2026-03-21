!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!input_in = !torch.vtensor<[$$input_input_shape$$], $$element_type$$>
!input_c = !torch.vtensor<[$$input_compute_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %input_raw: !input_in)
      attributes {torch.assume_strict_symbolic_shapes} {
$$input_adapter$$    $$extra_arg_decls$$
    %result = $$torch_op$$ %$$input_name$$$$extra_args$$ : !input_c$$extra_arg_types$$ -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
