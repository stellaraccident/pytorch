// Batched matrix multiply template.
//
// Placeholders: element_type, func_name, mat1_shape, mat2_shape, out_shape

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %mat1: !torch.vtensor<[$$mat1_shape$$], $$element_type$$>,
      %mat2: !torch.vtensor<[$$mat2_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %result = torch.aten.bmm %mat1, %mat2 : !torch.vtensor<[$$mat1_shape$$], $$element_type$$>, !torch.vtensor<[$$mat2_shape$$], $$element_type$$> -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
