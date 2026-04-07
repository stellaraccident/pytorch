#include <c10/core/Allocator.h>
#include <c10/pyre/impl/PyreStorage.h>

namespace c10::pyre::impl {
namespace host_allocator_registration {
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &c10::pyre::PyreStorageAllocator::hostAllocator())
} // namespace host_allocator_registration
namespace gpu_allocator_registration {
REGISTER_ALLOCATOR(c10::DeviceType::HIP, &c10::pyre::PyreStorageAllocator::gpuAllocator())
} // namespace gpu_allocator_registration
} // namespace c10::pyre::impl
