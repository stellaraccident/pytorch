#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace c10::pyre {
namespace {

size_t pyreDeviceTypeSlot(DeviceType type) {
  switch (type) {
    case DeviceType::PrivateUse1:
      return 0;
    case DeviceType::HIP:
      return 1;
    default:
      TORCH_CHECK(false, "pyre: unsupported device type ", type);
  }
}

} // namespace

PyreRuntime& PyreRuntime::get() {
  static PyreRuntime instance;
  return instance;
}

PyreRuntime::PyreRuntime()
    : host_allocator_(pyre_host_allocator_system()) {
  initialize();
}

PyreRuntime::~PyreRuntime() {
  for (auto& devices : devices_by_type_) {
    devices.clear();
  }
  if (gpu_initialized_) {
    pyre_log_status(pyre_gpu_shutdown(), "shutting down pyre GPU runtime");
  }
  pyre_log_status(pyre_cpu_shutdown(), "shutting down pyre CPU runtime");
}

void PyreRuntime::initialize() {
  PYRE_CHECK_OK(pyre_cpu_initialize(/*flags=*/0));

  int count = 0;
  PYRE_CHECK_OK(pyre_cpu_device_count(&count));
  TORCH_CHECK(count > 0, "pyre: CPU runtime initialized with no devices");

  auto& host_devices = devicesForType(DeviceType::PrivateUse1);
  host_devices.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    pyre_device_t device = nullptr;
    PYRE_CHECK_OK(pyre_cpu_device_get(i, &device));
    host_devices.push_back(std::make_unique<PyreDevice>(device));
  }

  pyre_status_t gpu_status = pyre_gpu_initialize(/*flags=*/0);
  if (!pyre_status_is_ok(gpu_status)) {
    auto message = formatPyreStatus(gpu_status);
    auto code = pyre_status_code(gpu_status);
    pyre_status_ignore(gpu_status);
    if (code == PYRE_STATUS_UNAVAILABLE) {
      return;
    }
    host_devices.clear();
    pyre_log_status(
        pyre_cpu_shutdown(),
        "rolling back pyre CPU runtime after GPU init failure");
    TORCH_CHECK(false, "pyre: failed to initialize GPU runtime: ", message);
  }
  gpu_initialized_ = true;

  int gpu_count = 0;
  PYRE_CHECK_OK(pyre_gpu_device_count(&gpu_count));

  auto& gpu_devices = devicesForType(DeviceType::HIP);
  gpu_devices.reserve(static_cast<size_t>(gpu_count));
  for (int i = 0; i < gpu_count; ++i) {
    pyre_device_t device = nullptr;
    PYRE_CHECK_OK(pyre_gpu_device_get(i, &device));
    gpu_devices.push_back(std::make_unique<PyreDevice>(device));
  }
}

std::vector<std::unique_ptr<PyreDevice>>& PyreRuntime::devicesForType(
    DeviceType type) {
  return devices_by_type_[pyreDeviceTypeSlot(type)];
}

const std::vector<std::unique_ptr<PyreDevice>>& PyreRuntime::devicesForType(
    DeviceType type) const {
  return devices_by_type_[pyreDeviceTypeSlot(type)];
}

PyreDevice* PyreRuntime::device(DeviceType type, DeviceIndex index) {
  const auto& devices = devicesForType(type);
  TORCH_CHECK(
      index >= 0 && static_cast<size_t>(index) < devices.size(),
      "pyre: invalid ", type, " device index ", index,
      ", expected 0..", devices.empty() ? 0 : devices.size() - 1);
  return devices[static_cast<size_t>(index)].get();
}

int32_t PyreRuntime::deviceCount(DeviceType type) const {
  return static_cast<int32_t>(devicesForType(type).size());
}

PyreDevice* PyreRuntime::device(DeviceIndex index) {
  return device(DeviceType::PrivateUse1, index);
}

int32_t PyreRuntime::deviceCount() const {
  return deviceCount(DeviceType::PrivateUse1);
}

} // namespace c10::pyre
