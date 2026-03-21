!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!weight = !torch.vtensor<[$$weight_shape$$], $$element_type$$>
!indices = !torch.vtensor<[$$indices_shape$$], si64>

module @module {
  func.func @$$func_name$$(
      %out_: !out_t,
      %weight: !weight,
      %indices: !indices
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %padding_idx = torch.constant.int -1
    %scale = torch.constant.bool false
    %sparse = torch.constant.bool false
    %result = torch.aten.embedding %weight, %indices, %padding_idx, %scale, %sparse : !weight, !indices, !torch.int, !torch.bool, !torch.bool -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
