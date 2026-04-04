#pragma once

// Centralized dtype mapping between PyTorch ScalarType, torch-mlir MLIR types,
// and pyre buffer-view element types.

#include <pyre_runtime.h>

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

// PyTorch ScalarType → pyre buffer-view element type.
inline pyre_element_type_t scalarTypeToHalElement(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float: return PYRE_ELEMENT_TYPE_FLOAT_32;
    case c10::ScalarType::Double: return PYRE_ELEMENT_TYPE_FLOAT_64;
    case c10::ScalarType::Half: return PYRE_ELEMENT_TYPE_FLOAT_16;
    case c10::ScalarType::BFloat16: return PYRE_ELEMENT_TYPE_BFLOAT_16;
    case c10::ScalarType::Int: return PYRE_ELEMENT_TYPE_INT_32;
    case c10::ScalarType::Long: return PYRE_ELEMENT_TYPE_INT_64;
    case c10::ScalarType::Short: return PYRE_ELEMENT_TYPE_INT_16;
    case c10::ScalarType::Byte: return PYRE_ELEMENT_TYPE_UINT_8;
    case c10::ScalarType::Char: return PYRE_ELEMENT_TYPE_SINT_8;
    case c10::ScalarType::Bool: return PYRE_ELEMENT_TYPE_BOOL_8;
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

// Element size → native IREE signless integer type string.
// For data movement kernels (copy, fill) that use opaque bit-width types.
inline const char* elementSizeToNativeInt(int64_t element_size) {
  switch (element_size) {
    case 1: return "i8";
    case 2: return "i16";
    case 4: return "i32";
    case 8: return "i64";
    default:
      TORCH_CHECK(false, "pyre: unsupported element size ", element_size);
  }
}

} // namespace at::pyre
