#include <c10/pyre/impl/PyreGuardImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10::pyre::impl {

C10_REGISTER_GUARD_IMPL(PrivateUse1, PyreHostGuardImpl)
C10_REGISTER_GUARD_IMPL(HIP, PyreGpuGuardImpl)

} // namespace c10::pyre::impl
