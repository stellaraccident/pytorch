// addmm with mat2 transposed: bias + mat1 @ mat2.T
// mat2 is passed in its original (un-transposed) layout.
// The transpose happens inside the kernel via torch.aten.t.
//
// Placeholders:
//   $$element_type$$    — torch-mlir element type
//   $$func_name$$       — entry point name
//   $$bias_shape$$      — bias shape
//   $$mat1_shape$$      — mat1 shape [M,K]
//   $$mat2_orig_shape$$ — mat2 original shape [N,K] (before transpose)
//   $$out_shape$$       — output shape [M,N]

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %bias: !torch.vtensor<[$$bias_shape$$], $$element_type$$>,
      %mat1: !torch.vtensor<[$$mat1_shape$$], $$element_type$$>,
      %mat2_orig: !torch.vtensor<[$$mat2_orig_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %mat2 = torch.aten.t %mat2_orig : !torch.vtensor<[$$mat2_orig_shape$$], $$element_type$$> -> !torch.vtensor<[$$mat2_t_shape$$], $$element_type$$>
    %one = torch.constant.int 1
    %result = torch.aten.addmm %bias, %mat1, %mat2, %one, %one : !torch.vtensor<[$$bias_shape$$], $$element_type$$>, !torch.vtensor<[$$mat1_shape$$], $$element_type$$>, !torch.vtensor<[$$mat2_t_shape$$], $$element_type$$>, !torch.int, !torch.int -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
