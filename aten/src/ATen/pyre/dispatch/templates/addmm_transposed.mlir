// addmm with mat2 transposed: bias + mat1 @ mat2.T
// mat2 is passed in its original (un-transposed) layout.
// The transpose happens inside the kernel via torch.aten.t.
//
// Placeholders: element_type, func_name, bias_shape, mat1_shape,
//   mat2_orig_shape, mat2_t_shape, out_shape,
//   beta_decl, alpha_decl, beta_type, alpha_type

!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!bias = !torch.vtensor<[$$bias_shape$$], $$element_type$$>
!mat1 = !torch.vtensor<[$$mat1_shape$$], $$element_type$$>
!mat2_orig = !torch.vtensor<[$$mat2_orig_shape$$], $$element_type$$>
!mat2_t = !torch.vtensor<[$$mat2_t_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %bias: !bias, %mat1: !mat1, %mat2_orig: !mat2_orig)
      attributes {torch.assume_strict_symbolic_shapes} {
    %mat2 = torch.aten.t %mat2_orig : !mat2_orig -> !mat2_t
    $$beta_decl$$
    $$alpha_decl$$
    %result = torch.aten.addmm %bias, %mat1, %mat2, %beta, %alpha : !bias, !mat1, !mat2_t, $$beta_type$$, $$alpha_type$$ -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
