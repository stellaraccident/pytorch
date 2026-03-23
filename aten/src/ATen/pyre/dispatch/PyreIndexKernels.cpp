#include <ATen/pyre/dispatch/PyreAlgorithmicKernels.h>

namespace at::pyre {
namespace {

// ---------------------------------------------------------------------------
// index.Tensor fragments
// ---------------------------------------------------------------------------

constexpr std::string_view kIndexHeader = R"(
module @module {
  func.func @$$FUNC$$(
      %out_: !torch.tensor<[$$OUT_SHAPE$$], $$ELT$$>,
      %self: !torch.vtensor<[$$SELF_SHAPE$$], $$ELT$$>)";

constexpr std::string_view kIndexInput = R"(,
      %idx$$IDX$$: !torch.vtensor<[$$IDX_SHAPE$$], si64>)";

constexpr std::string_view kIndexNoneDecl = R"(
    %none$$IDX$$ = torch.constant.none
    %opt$$IDX$$ = torch.derefine %none$$IDX$$ : !torch.none to !torch.optional<vtensor>)";

constexpr std::string_view kIndexTensorDecl = R"(
    %opt$$IDX$$ = torch.derefine %idx$$TIDX$$ : !torch.vtensor<[$$IDX_SHAPE$$], si64> to !torch.optional<vtensor>)";

constexpr std::string_view kIndexBody = R"(
  ) attributes {torch.assume_strict_symbolic_shapes} {$$OPT_DECLS$$
    %list = torch.prim.ListConstruct $$OPT_NAMES$$ : ($$OPT_TYPES$$) -> !torch.list<optional<vtensor>>
    %result = torch.aten.index.Tensor %self, %list : !torch.vtensor<[$$SELF_SHAPE$$], $$ELT$$>, !torch.list<optional<vtensor>> -> !torch.vtensor<[$$OUT_SHAPE$$], $$ELT$$>
    torch.overwrite.tensor.contents %result overwrites %out_ : !torch.vtensor<[$$OUT_SHAPE$$], $$ELT$$>, !torch.tensor<[$$OUT_SHAPE$$], $$ELT$$>
    return
  }
}
)";

// ---------------------------------------------------------------------------
// index_put (functional) fragments
// ---------------------------------------------------------------------------

constexpr std::string_view kIndexPutHeader = R"(
module @module {
  func.func @$$FUNC$$(
      %out_: !torch.tensor<[$$SELF_LOG$$], $$ELT$$>,
      %self_raw: $$SELF_PHYS_T$$)";

constexpr std::string_view kIndexPutInput = R"(,
      %idx$$IDX$$: !torch.vtensor<[$$IDX_SHAPE$$], si64>)";

constexpr std::string_view kIndexPutBody = R"(,
      %values_raw: $$VALS_PHYS_T$$
  ) attributes {torch.assume_strict_symbolic_shapes} {
$$SELF_ADAPT$$$$VALS_ADAPT$$$$OPT_DECLS$$
    %accum = torch.constant.bool $$ACCUM$$
    %list = torch.prim.ListConstruct $$OPT_NAMES$$ : ($$OPT_TYPES$$) -> !torch.list<optional<vtensor>>
    %result = torch.aten.index_put %$$SELF_NAME$$, %list, %$$VALS_NAME$$, %accum : $$SELF_LOG_T$$, !torch.list<optional<vtensor>>, $$VALS_LOG_T$$, !torch.bool -> $$SELF_LOG_T$$
    torch.overwrite.tensor.contents %result overwrites %out_ : $$SELF_LOG_T$$, !torch.tensor<[$$SELF_LOG$$], $$ELT$$>
    return
  }
}
)";

// ---------------------------------------------------------------------------
// index_put_ (in-place) fragments — reads self via copy.to_vtensor
// ---------------------------------------------------------------------------

constexpr std::string_view kIndexPutInplaceHeader = R"(
module @module {
  func.func @$$FUNC$$(
      %out_: !torch.tensor<[$$SELF_LOG$$], $$ELT$$>)";

constexpr std::string_view kIndexPutInplaceBody = R"(,
      %values_raw: $$VALS_PHYS_T$$
  ) attributes {torch.assume_strict_symbolic_shapes} {
    %self = torch.copy.to_vtensor %out_ : $$SELF_LOG_T$$
$$VALS_ADAPT$$$$OPT_DECLS$$
    %accum = torch.constant.bool $$ACCUM$$
    %list = torch.prim.ListConstruct $$OPT_NAMES$$ : ($$OPT_TYPES$$) -> !torch.list<optional<vtensor>>
    %result = torch.aten.index_put %self, %list, %$$VALS_NAME$$, %accum : $$SELF_LOG_T$$, !torch.list<optional<vtensor>>, $$VALS_LOG_T$$, !torch.bool -> $$SELF_LOG_T$$
    torch.overwrite.tensor.contents %result overwrites %out_ : $$SELF_LOG_T$$, !torch.tensor<[$$SELF_LOG$$], $$ELT$$>
    return
  }
}
)";

} // namespace

// ---------------------------------------------------------------------------
// Fragment accessors
// ---------------------------------------------------------------------------

PyreKernelAsmFragments& indexFragments() {
  static PyreKernelAsmFragments frags(
      {kIndexHeader, kIndexInput, kIndexNoneDecl, kIndexTensorDecl, kIndexBody});
  return frags;
}

PyreKernelAsmFragments& indexPutFragments() {
  static PyreKernelAsmFragments frags(
      {kIndexPutHeader, kIndexPutInput, kIndexPutBody});
  return frags;
}

PyreKernelAsmFragments& indexPutInplaceFragments() {
  static PyreKernelAsmFragments frags(
      {kIndexPutInplaceHeader, kIndexPutInput, kIndexPutInplaceBody});
  return frags;
}

} // namespace at::pyre
