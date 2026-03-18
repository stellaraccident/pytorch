#pragma once

// PyreHostGuardImpl: Device guard for the host backend (PrivateUse1).
//
// When GPU (HIP) support arrives, a separate PyreGpuGuardImpl will
// be registered for DeviceType::HIP.

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreStream.h>

namespace c10::pyre::impl {

// Thread-local current device index for the host backend.
inline thread_local DeviceIndex current_host_device_index = 0;

struct PyreHostGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  PyreHostGuardImpl() = default;

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  Device exchangeDevice(Device d) const override {
    auto old = Device(c10::DeviceType::PrivateUse1, current_host_device_index);
    current_host_device_index = d.index();
    return old;
  }

  Device getDevice() const override {
    return Device(c10::DeviceType::PrivateUse1, current_host_device_index);
  }

  void setDevice(Device d) const override {
    TORCH_CHECK(
        d.index() >= 0 && d.index() < PyreDevice::deviceCount(),
        "pyre: invalid host device index ", d.index());
    current_host_device_index = d.index();
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    current_host_device_index = d.index();
  }

  Stream getStream(Device d) const noexcept override {
    return getCurrentHostStream(d.index());
  }

  Stream getDefaultStream(Device d) const override {
    return getDefaultHostStream(d.index());
  }

  Stream getNewStream(Device d, int priority) const override {
    auto* device = PyreDevice::get(d.index());
    bool high = priority > 0;
    StreamId id = device->getStreamFromPool(high);
    return Stream(Stream::UNSAFE, d, id);
  }

  Stream exchangeStream(Stream s) const noexcept override {
    auto old = getCurrentHostStream(s.device_index());
    setCurrentHostStream(s);
    return old;
  }

  DeviceIndex deviceCount() const noexcept override {
    return static_cast<DeviceIndex>(PyreDevice::deviceCount());
  }
};

} // namespace c10::pyre::impl
