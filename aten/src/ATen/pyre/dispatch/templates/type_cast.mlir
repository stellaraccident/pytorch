// Type cast template (_to_copy with dtype change).
//
// Placeholders: in_element_type, out_element_type, func_name,
//   input_shape, out_shape, dtype_const

!out_t = !torch.tensor<[$$out_shape$$], $$out_element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$out_element_type$$>
!input = !torch.vtensor<[$$input_shape$$], $$in_element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %input: !input)
      attributes {torch.assume_strict_symbolic_shapes} {
    $$dtype_const$$
    %none = torch.constant.none
    %false = torch.constant.bool false
    %result = torch.aten._to_copy %input, %dtype, %none, %none, %none, %false, %none : !input, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
