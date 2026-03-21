!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!mat1_in = !torch.vtensor<[$$mat1_input_shape$$], $$element_type$$>
!mat1_c = !torch.vtensor<[$$mat1_compute_shape$$], $$element_type$$>
!mat2_in = !torch.vtensor<[$$mat2_input_shape$$], $$element_type$$>
!mat2_c = !torch.vtensor<[$$mat2_compute_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %mat1_raw: !mat1_in, %mat2_raw: !mat2_in)
      attributes {torch.assume_strict_symbolic_shapes} {
$$mat1_adapter$$$$mat2_adapter$$    %result = torch.aten.mm %$$mat1_name$$, %$$mat2_name$$ : !mat1_c, !mat2_c -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
