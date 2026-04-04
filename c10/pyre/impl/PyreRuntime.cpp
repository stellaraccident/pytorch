#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace c10::pyre {

PyreRuntime& PyreRuntime::get() {
  static PyreRuntime instance;
  return instance;
}

PyreRuntime::PyreRuntime()
    : host_allocator_(pyre_host_allocator_system()) {
  initialize();
}

PyreRuntime::~PyreRuntime() {
  devices_.clear();
  pyre_log_status(pyre_cpu_shutdown(), "shutting down pyre CPU runtime");
}

void PyreRuntime::initialize() {
  PYRE_CHECK_OK(pyre_cpu_initialize(/*flags=*/0));

  int count = 0;
  PYRE_CHECK_OK(pyre_cpu_device_count(&count));
  TORCH_CHECK(count > 0, "pyre: CPU runtime initialized with no devices");

  devices_.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    pyre_device_t device = nullptr;
    PYRE_CHECK_OK(pyre_cpu_device_get(i, &device));
    devices_.push_back(std::make_unique<PyreDevice>(device));
  }
}

PyreDevice* PyreRuntime::device(DeviceIndex index) {
  TORCH_CHECK(
      index >= 0 && static_cast<size_t>(index) < devices_.size(),
      "pyre: invalid device index ", index,
      ", expected 0..", devices_.size() - 1);
  return devices_[static_cast<size_t>(index)].get();
}

int32_t PyreRuntime::deviceCount() const {
  return static_cast<int32_t>(devices_.size());
}

} // namespace c10::pyre
