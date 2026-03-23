#include <ATen/pyre/dispatch/PyreAlgorithmicKernels.h>

namespace at::pyre {
namespace {

constexpr std::string_view kCatHeader = R"(
module @module {
  func.func @$$FUNC$$(
      %out_: !torch.tensor<[$$OUT_SHAPE$$], $$ELT$$>)";

constexpr std::string_view kCatInput = R"(,
      %input$$IDX$$: !torch.vtensor<[$$INPUT_SHAPE$$], $$ELT$$>)";

constexpr std::string_view kCatBody = R"(
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %inputs = torch.prim.ListConstruct $$INPUT_NAMES$$ : ($$INPUT_TYPES$$) -> !torch.list<vtensor>
    %dim = torch.constant.int $$DIM$$
    %result = torch.aten.cat %inputs, %dim : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[$$OUT_SHAPE$$], $$ELT$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$OUT_SHAPE$$], $$ELT$$>, !torch.tensor<[$$OUT_SHAPE$$], $$ELT$$>
    return
  }
}
)";

} // namespace

PyreKernelAsmFragments& catFragments() {
  static PyreKernelAsmFragments frags({kCatHeader, kCatInput, kCatBody});
  return frags;
}

} // namespace at::pyre
