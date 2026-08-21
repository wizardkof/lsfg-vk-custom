/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"

#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lsfgvk::backend {

    class [[gnu::visibility("default")]] ContextImpl;
    class [[gnu::visibility("default")]] InstanceImpl;

    using Context = ContextImpl;

    ///
    /// Primitive exception class that deliveres a detailed error message
    ///
    class [[gnu::visibility("default")]] error : public std::runtime_error {
    public:
        ///
        /// Construct an error
        ///
        /// @param msg Error message.
        /// @param inner Inner exception.
        ///
        explicit error(const std::string &msg, const std::exception &inner);

        ///
        /// Construct an error
        ///
        /// @param msg Error message.
        ///
        explicit error(const std::string &msg);

        error(const error &) = default;
        error &operator=(const error &) = default;
        error(error &&) = default;
        error &operator=(error &&) = default;
        ~error() override;
    };

    /// Function type for picking a device based on its stable identity.
    using DevicePicker = std::function<bool(const vk::PhysicalDeviceIdentity& identity)>;
    /// One-shot observer for the complete backend-visible device inventory.
    using DeviceEnumerationObserver = std::function<void(
        const std::vector<vk::PhysicalDeviceSnapshot>& devices)>;

    ///
    /// Main entry point of the library
    ///
    class [[gnu::visibility("default")]] Instance {
    public:
        ///
        /// Create a lsfg-vk instance
        ///
        /// @param devicePicker Function that picks a physical device based on some identifiers.
        /// @param shaderDllPath Path to the Lossless.dll file to load shaders from.
        /// @param allowLowPrecision Whether to load low-precision (FP16) shaders if supported.
        ///
        /// @throws backend::error on failure
        ///
        Instance(
            const DevicePicker& devicePicker,
            const std::filesystem::path& shaderDllPath,
            bool allowLowPrecision,
            const DeviceEnumerationObserver& deviceObserver = {}
        );

        /// Identity of the physical device selected for frame generation.
        [[nodiscard]] const vk::PhysicalDeviceIdentity& deviceIdentity() const;
        /// Complete physical-device inventory seen by the backend VkInstance.
        [[nodiscard]] const std::vector<vk::PhysicalDeviceSnapshot>& visibleDevices() const;
        /// Narrow Vulkan endpoint used by the P3D runtime cross-device control channel.
        /// Handles remain owned by backend::Instance.
        [[nodiscard]] vk::RuntimeExchangeEndpoint runtimeExchangeEndpoint() const;

        /// Run the real LSFG prepass against one frame already resident on the
        /// backend device. This is a diagnostic-only, zero-generation path.
        void validateRuntimePrepass(VkImage transportImage, VkExtent2D extent,
            VkFormat transportFormat, uint64_t transportModifier,
            float flow, bool perf);

        ///
        /// Open a frame generation context.
        ///
        /// The VkFormat of the exchanged images is inferred from whether hdr is true or false:
        /// - false: VK_FORMAT_R8G8B8A8_UNORM
        /// - true: VK_FORMAT_R16G16B16A16_SFLOAT
        ///
        /// The application and library must keep track of the frame index. When the next frame
        /// is ready, signal the syncFd with one increment (with the first trigger being 1).
        /// Each generated frame will increment the semaphore by one:
        /// - Application signals 1 -> Start generating with (curr, next) source images
        /// - Library signals 1 -> First frame between (curr, next) is ready
        /// - Library signals N -> N-th frame between (curr, next) is ready
        /// - Application signals N+1 -> Start generating with (next, curr) source images
        ///
        /// @param sourceImages Pair of exported source images; ownership is consumed.
        /// @param destImages Vector of exported output images; ownership is consumed.
        /// @param syncFd File descriptor for the timeline semaphore used for synchronization.
        /// @param flow Motion flow factor.
        /// @param perf Whether to enable performance mode.
        ///
        /// @throws backend::error on failure
        ///
        Context& openContext(
            std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
            std::vector<vk::ExternalImage> destImages,
            int syncFd,
            float flow, bool perf
        );

        ///
        /// Schedule a new set of generated frames.
        ///
        /// @param context Context to use.
        /// @throws backend::error on failure
        ///
        void scheduleFrames(Context& context);

        ///
        /// Schedule zero or more generated frames using explicit interpolation timestamps.
        ///
        /// The context still processes the real source frame when timestamps is empty,
        /// keeping temporal history and synchronization in step with the application.
        ///
        /// @param context Context to use.
        /// @param timestamps Strictly increasing normalized timestamps in the range (0, 1).
        /// @throws backend::error on failure.
        ///
        void scheduleFrames(
            Context& context,
            const std::vector<float>& timestamps
        );

        ///
        /// Close a frame generation context
        ///
        /// @param context Context to close.
        ///
        void closeContext(const Context& context);

        // Non-copyable and non-movable
        Instance(const Instance&) = delete;
        Instance& operator=(const Instance&) = delete;
        Instance(Instance&&) = delete;
        Instance& operator=(Instance&&) = delete;
        virtual ~Instance();
    private:
        std::unique_ptr<InstanceImpl> m_impl;

        std::vector<std::unique_ptr<Context>> m_contexts;
    };

    ///
    /// Make all lsfg-vk instances leaking.
    /// This is to workaround a bug in the Vulkan loader, which
    /// makes it impossible to destroy Vulkan instances and devices.
    ///
    void makeLeaking();

}
