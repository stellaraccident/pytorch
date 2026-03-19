// Matrix multiply: out = mat1 @ mat2
//
// Placeholders:
//   $$element_type$$  — torch-mlir element type
//   $$func_name$$     — entry point name
//   $$mat1_shape$$    — mat1 shape [M,K]
//   $$mat2_shape$$    — mat2 shape [K,N]
//   $$out_shape$$     — output shape [M,N]

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %mat1: !torch.vtensor<[$$mat1_shape$$], $$element_type$$>,
      %mat2: !torch.vtensor<[$$mat2_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %result = torch.aten.mm %mat1, %mat2 : !torch.vtensor<[$$mat1_shape$$], $$element_type$$>, !torch.vtensor<[$$mat2_shape$$], $$element_type$$> -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
