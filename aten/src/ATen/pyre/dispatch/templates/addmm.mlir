// addmm: beta * bias + alpha * (mat1 @ mat2)
// F.linear dispatches to this with beta=1, alpha=1.
//
// Placeholders:
//   $$element_type$$  — torch-mlir element type
//   $$func_name$$     — entry point name
//   $$bias_shape$$    — bias shape (e.g., "?")
//   $$mat1_shape$$    — mat1 shape (e.g., "?,?")
//   $$mat2_shape$$    — mat2 shape (e.g., "?,?")
//   $$out_shape$$     — output shape (e.g., "?,?")
//   $$beta_decl$$     — beta constant declaration
//   $$alpha_decl$$    — alpha constant declaration
//   $$beta_type$$     — !torch.int or !torch.float
//   $$alpha_type$$    — !torch.int or !torch.float

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %bias: !torch.vtensor<[$$bias_shape$$], $$element_type$$>,
      %mat1: !torch.vtensor<[$$mat1_shape$$], $$element_type$$>,
      %mat2: !torch.vtensor<[$$mat2_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    $$beta_decl$$
    $$alpha_decl$$
    %result = torch.aten.addmm %bias, %mat1, %mat2, %beta, %alpha : !torch.vtensor<[$$bias_shape$$], $$element_type$$>, !torch.vtensor<[$$mat1_shape$$], $$element_type$$>, !torch.vtensor<[$$mat2_shape$$], $$element_type$$>, $$beta_type$$, $$alpha_type$$ -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
