#include <c10/core/Allocator.h>
#include <c10/pyre/impl/PyreStorage.h>

// Register the host device allocator for PrivateUse1.
// When GPU (HIP) support arrives, a second registration will be added
// with DEVICE_LOCAL memory params for DeviceType::HIP.
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &c10::pyre::PyreStorageAllocator::hostAllocator());
