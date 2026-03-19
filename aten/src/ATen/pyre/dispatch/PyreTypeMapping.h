#pragma once

// Centralized dtype mapping between PyTorch ScalarType, torch-mlir MLIR types,
// and IREE HAL element types.

#include <iree/hal/api.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

namespace at::pyre {

// PyTorch ScalarType → torch-mlir MLIR element type string.
// torch-mlir uses "f32", "f64", "si32", "si64" (signed integers).
inline const char* scalarTypeToTorchMlir(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float: return "f32";
    case c10::ScalarType::Double: return "f64";
    case c10::ScalarType::Half: return "f16";
    case c10::ScalarType::BFloat16: return "bf16";
    case c10::ScalarType::Int: return "si32";
    case c10::ScalarType::Long: return "si64";
    case c10::ScalarType::Short: return "si16";
    case c10::ScalarType::Byte: return "ui8";
    case c10::ScalarType::Char: return "si8";
    case c10::ScalarType::Bool: return "i1";
    default:
      TORCH_CHECK(false, "pyre: unsupported dtype: ", c10::toString(dtype));
  }
}

// PyTorch ScalarType → IREE HAL element type.
inline iree_hal_element_type_t scalarTypeToHalElement(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float: return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case c10::ScalarType::Double: return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    case c10::ScalarType::Half: return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case c10::ScalarType::BFloat16: return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    case c10::ScalarType::Int: return IREE_HAL_ELEMENT_TYPE_INT_32;
    case c10::ScalarType::Long: return IREE_HAL_ELEMENT_TYPE_INT_64;
    case c10::ScalarType::Short: return IREE_HAL_ELEMENT_TYPE_INT_16;
    case c10::ScalarType::Byte: return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case c10::ScalarType::Char: return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case c10::ScalarType::Bool: return IREE_HAL_ELEMENT_TYPE_BOOL_8;
    default:
      TORCH_CHECK(false, "pyre: unsupported dtype for HAL: ",
                  c10::toString(dtype));
  }
}

// Whether a dtype uses floating-point arithmetic ops in MLIR.
inline bool isFloatDtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
    case c10::ScalarType::Double:
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      return true;
    default:
      return false;
  }
}

} // namespace at::pyre
