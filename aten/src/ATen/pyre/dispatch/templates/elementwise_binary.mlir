// Binary elementwise template (add, sub, mul, div).
// Uses torch-mlir calling convention matching fusilli's pattern.
//
// Placeholders: element_type, func_name, lhs_shape, rhs_shape, out_shape,
//   torch_op, extra_args, extra_arg_decls, extra_arg_types

!out_t = !torch.tensor<[$$out_shape$$], $$element_type$$>
!out_v = !torch.vtensor<[$$out_shape$$], $$element_type$$>
!lhs = !torch.vtensor<[$$lhs_shape$$], $$element_type$$>
!rhs = !torch.vtensor<[$$rhs_shape$$], $$element_type$$>

module @module {
  func.func @$$func_name$$(%out_: !out_t, %lhs: !lhs, %rhs: !rhs)
      attributes {torch.assume_strict_symbolic_shapes} {
    $$extra_arg_decls$$
    %result = $$torch_op$$ %lhs, %rhs$$extra_args$$ : !lhs, !rhs$$extra_arg_types$$ -> !out_v
    torch.overwrite.tensor.contents %result overwrites %out_ : !out_v, !out_t
    return
  }
}
