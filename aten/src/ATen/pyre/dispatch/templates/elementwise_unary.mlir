// Unary elementwise template (neg, abs, relu).
// Uses torch-mlir calling convention matching fusilli's pattern.
//
// Placeholders:
//   $$element_type$$   — torch element type (f32, f64, si32, si64)
//   $$func_name$$      — entry point name
//   $$input_shape$$    — input shape comma-separated
//   $$out_shape$$      — output shape comma-separated
//   $$torch_op$$       — torch dialect op (torch.aten.neg, etc.)

module @module {
  func.func @$$func_name$$(
      %out_: !torch.tensor<[$$out_shape$$], $$element_type$$>,
      %input: !torch.vtensor<[$$input_shape$$], $$element_type$$>
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %result = $$torch_op$$ %input : !torch.vtensor<[$$input_shape$$], $$element_type$$> -> !torch.vtensor<[$$out_shape$$], $$element_type$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$out_shape$$], $$element_type$$>, !torch.tensor<[$$out_shape$$], $$element_type$$>
    return
  }
}
