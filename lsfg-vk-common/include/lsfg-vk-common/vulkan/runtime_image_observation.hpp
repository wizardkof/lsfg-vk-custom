/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "runtime_exchange_channel.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <vulkan/vulkan_core.h>

namespace vk {

    /// Borrowed image description for one asynchronous observation submission.
    /// The lifetime token is locked by submit() and retained until retirement.
    struct RuntimeImageObservationDescriptor {
        VkImage image{VK_NULL_HANDLE};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkExtent2D extent{};
        uint32_t queueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
        std::weak_ptr<const uint8_t> lifetime;
    };

    struct RuntimeImageObservationSessionInfo {
        uint32_t slotCount{};
        VkDeviceSize maxByteCount{};
    };

    struct RuntimeImageObservationTicket {
        uint32_t slotIndex{UINT32_MAX};
        uint64_t epoch{};

        [[nodiscard]] bool valid() const noexcept {
            return slotIndex != UINT32_MAX && epoch != 0;
        }
    };

    enum class RuntimeImageObservationSubmitStatus : uint8_t {
        SUBMITTED,
        TEMPORARILY_BLOCKED
    };

    struct RuntimeImageObservationSubmitResult {
        RuntimeImageObservationSubmitStatus status{
            RuntimeImageObservationSubmitStatus::TEMPORARILY_BLOCKED};
        RuntimeImageObservationTicket ticket{};
    };

    struct RuntimeImageObservation {
        VkImage image{VK_NULL_HANDLE};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkExtent2D extent{};
        uint32_t queueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
        size_t byteCount{};
        size_t nonzeroByteCount{};
        uint64_t checksum{};
    };

    enum class RuntimeImageObservationRetireStatus : uint8_t {
        NOT_READY,
        RETIRED
    };

    struct RuntimeImageObservationRetireResult {
        RuntimeImageObservationRetireStatus status{
            RuntimeImageObservationRetireStatus::NOT_READY};
        std::optional<RuntimeImageObservation> observation;
    };

    /// Exact tightly packed byte count supported by the observation copy.
    /// Only uncompressed four-byte RGBA/BGRA formats are accepted.
    [[nodiscard]] std::optional<VkDeviceSize> runtimeImageObservationByteCount(
        VkFormat format, VkExtent2D extent) noexcept;

    /// Bounded, reusable, nonblocking GPU image observation session.
    ///
    /// The caller must retire every accepted ticket before destroying the
    /// session.  To preserve Vulkan resource and source-image lifetime without
    /// introducing a host wait, destruction with work still in flight abandons
    /// that bounded resource block instead of destroying GPU-visible objects.
    class RuntimeImageObservationSession {
    public:
        RuntimeImageObservationSession() noexcept = default;
        RuntimeImageObservationSession(const RuntimeImageObservationSession&) = delete;
        RuntimeImageObservationSession& operator=(const RuntimeImageObservationSession&) = delete;
        RuntimeImageObservationSession(RuntimeImageObservationSession&&) noexcept;
        RuntimeImageObservationSession& operator=(RuntimeImageObservationSession&&) noexcept;
        ~RuntimeImageObservationSession();

        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] uint32_t slotCount() const noexcept;
        [[nodiscard]] uint32_t inFlightCount() const noexcept;
        [[nodiscard]] VkDeviceSize maxByteCount() const noexcept;

        /// Validate and enqueue one image-to-staging copy.  This never waits on
        /// the host.  When every slot is occupied, it returns
        /// TEMPORARILY_BLOCKED without resetting or submitting anything.
        [[nodiscard]] RuntimeImageObservationSubmitResult submit(
            const RuntimeImageObservationDescriptor& descriptor);

        /// Poll exactly one accepted ticket with vkGetFenceStatus.  NOT_READY
        /// has no side effects; RETIRED computes byte count, nonzero-byte count,
        /// and FNV-1a from the completed staging contents and releases the slot.
        [[nodiscard]] RuntimeImageObservationRetireResult tryRetire(
            RuntimeImageObservationTicket ticket);

    private:
        struct Impl;
        explicit RuntimeImageObservationSession(std::unique_ptr<Impl>) noexcept;
        void reset() noexcept;

        std::unique_ptr<Impl> impl;

        friend RuntimeImageObservationSession createRuntimeImageObservationSession(
            RuntimeExchangeEndpoint, RuntimeImageObservationSessionInfo);
    };

    [[nodiscard]] RuntimeImageObservationSession createRuntimeImageObservationSession(
        RuntimeExchangeEndpoint endpoint, RuntimeImageObservationSessionInfo info);

}
