#pragma once

// PyreOpJit: compiled kernel dispatch for individual ATen ops.
// Handles template expansion, compilation, caching, and invocation.

#include <ATen/Tensor.h>
#include <c10/core/Scalar.h>

namespace at::pyre {

bool jitAvailable();

// Binary ops (add, sub, mul, div).
at::Tensor jitBinaryOp(
    const at::Tensor& self,
    const at::Tensor& other,
    const std::string& func_name,
    const std::string& linalg_op);

// Add/sub with alpha parameter.
at::Tensor jitAddOp(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);

at::Tensor jitSubOp(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);

// Matrix multiply.
at::Tensor jitMmOp(
    const at::Tensor& self,
    const at::Tensor& other);

// addmm: beta * bias + alpha * (mat1 @ mat2)
at::Tensor jitAddmmOp(
    const at::Tensor& bias,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha);

// Unary ops.
at::Tensor jitUnaryOp(
    const at::Tensor& self,
    const std::string& func_name,
    const std::string& torch_op);

} // namespace at::pyre
