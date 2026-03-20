// Type cast template (_to_copy with dtype change).
//
// Placeholders: in_element_type, out_element_type, func_name,
//   input_shape, out_shape, dtype_const

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$out_element_type$$>,
      %input: !torch.vtensor<[$$input_shape$$], $$in_element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    $$dtype_const$$
    %none = torch.constant.none
    %false = torch.constant.bool false
    %result = torch.aten._to_copy %input, %dtype, %none, %none, %none, %false, %none : !torch.vtensor<[$$input_shape$$], $$in_element_type$$>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[$$out_shape$$], $$out_element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$out_element_type$$>, !torch.tensor<[$$out_shape$$], $$out_element_type$$>
    return
  }
}
