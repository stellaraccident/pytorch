// addmm: beta * bias + alpha * (mat1 @ mat2)
// F.linear dispatches to this with beta=1, alpha=1.
//
// Placeholders: element_type, func_name, bias_shape, mat1_shape,
//   mat2_shape, out_shape, beta_decl, alpha_decl, beta_type, alpha_type

!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!bias = !torch.vtensor<[$$bias_shape$$], $$element_type$$>
!mat1 = !torch.vtensor<[$$mat1_shape$$], $$element_type$$>
!mat2 = !torch.vtensor<[$$mat2_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %bias: !bias, %mat1: !mat1, %mat2: !mat2)
      attributes {torch.assume_strict_symbolic_shapes} {
    $$beta_decl$$
    $$alpha_decl$$
    %result = torch.aten.addmm %bias, %mat1, %mat2, %beta, %alpha : !bias, !mat1, !mat2, $$beta_type$$, $$alpha_type$$ -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
