!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!input = !torch.vtensor<[$$input_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(
      %out_: !out_t,
      %input: !input
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %dim = torch.constant.int $$dim$$
    %half_to_float = torch.constant.bool false
    %result = torch.aten.$$softmax_op$$ %input, %dim, %half_to_float : !input, !torch.int, !torch.bool -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
