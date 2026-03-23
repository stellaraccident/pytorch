!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!index = !torch.vtensor<[$$index_shape$$], si64>
!src = !torch.vtensor<[$$src_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(
      %out_: !out_t,
      %index: !index,
      %src: !src
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %self = torch.copy.to_vtensor %out_ : !out_v
    %dim = torch.constant.int $$dim$$
    %result = torch.aten.scatter_add %self, %dim, %index, %src : !out_v, !torch.int, !index, !src -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
