#pragma once

// Device guards for Pyre host/gpu frontends. Both delegate to the shared
// c10::pyre runtime/stream/allocator implementation.

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreStream.h>

namespace c10::pyre::impl {

template <c10::DeviceType DeviceTypeValue>
struct PyreGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  PyreGuardImpl() = default;

  c10::DeviceType type() const override {
    return DeviceTypeValue;
  }

  Device exchangeDevice(Device d) const override {
    auto old = getDevice();
    setCurrentPyreDeviceIndex(DeviceTypeValue, d.index());
    return old;
  }

  Device getDevice() const override {
    return Device(
        DeviceTypeValue, getCurrentPyreDeviceIndex(DeviceTypeValue));
  }

  void setDevice(Device d) const override {
    setCurrentPyreDeviceIndex(DeviceTypeValue, d.index());
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    setCurrentPyreDeviceIndex(DeviceTypeValue, d.index());
  }

  Stream getStream(Device d) const noexcept override {
    return getCurrentPyreStream(DeviceTypeValue, d.index());
  }

  Stream getDefaultStream(Device d) const override {
    return getDefaultPyreStream(DeviceTypeValue, d.index());
  }

  Stream getNewStream(Device d, int priority) const override {
    auto* device = PyreDevice::get(DeviceTypeValue, d.index());
    bool high = priority > 0;
    StreamId id = device->getStreamFromPool(high);
    return Stream(Stream::UNSAFE, d, id);
  }

  Stream exchangeStream(Stream s) const noexcept override {
    auto old = getCurrentPyreStream(DeviceTypeValue, s.device_index());
    setCurrentPyreStream(s);
    return old;
  }

  DeviceIndex deviceCount() const noexcept override {
    return static_cast<DeviceIndex>(PyreDevice::deviceCount(DeviceTypeValue));
  }
};

using PyreHostGuardImpl = PyreGuardImpl<c10::DeviceType::PrivateUse1>;
using PyreGpuGuardImpl = PyreGuardImpl<c10::DeviceType::HIP>;

} // namespace c10::pyre::impl
