!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!input = !torch.vtensor<[$$input_shape$$], $$element_type$$>
!index = !torch.vtensor<[$$index_shape$$], si64>

module @module {
  func.func @$$func_name$$(
      %out_: !out_t,
      %input: !input,
      %index: !index
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %dim = torch.constant.int $$dim$$
    %sparse_grad = torch.constant.bool false
    %result = torch.aten.gather %input, %dim, %index, %sparse_grad : !input, !torch.int, !index, !torch.bool -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
