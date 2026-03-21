!out_t = !torch.tensor<[$$out_size$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_size$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t) attributes {torch.assume_strict_symbolic_shapes} {
    $$start_decl$$
    $$end_decl$$
    $$step_decl$$
    %none = torch.constant.none
    %result = torch.aten.arange.start_step %start, %end, %step, %none, %none, %none, %none : $$scalar_type$$, $$scalar_type$$, $$scalar_type$$, !torch.none, !torch.none, !torch.none, !torch.none -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
