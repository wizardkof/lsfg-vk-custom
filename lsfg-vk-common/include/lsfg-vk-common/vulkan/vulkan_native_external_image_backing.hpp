#pragma once

#include "runtime_exchange_channel.hpp"

#include <vector>

namespace vk {
    struct VulkanNativeExternalImageDescriptor {
        ls::OwnedFd dmaBuf;
        VkFormat format{};
        VkExtent2D extent{};
        uint32_t fourcc{};
        uint64_t modifier{};
        std::vector<VkSubresourceLayout> planes;
        VkDeviceSize allocationSize{};
    };

    class VulkanNativeExternalImageBacking {
    public:
        VulkanNativeExternalImageBacking() noexcept = default;
        VulkanNativeExternalImageBacking(const VulkanNativeExternalImageBacking&) = delete;
        VulkanNativeExternalImageBacking& operator=(const VulkanNativeExternalImageBacking&) = delete;
        VulkanNativeExternalImageBacking(VulkanNativeExternalImageBacking&&) noexcept;
        VulkanNativeExternalImageBacking& operator=(VulkanNativeExternalImageBacking&&) noexcept;
        ~VulkanNativeExternalImageBacking();

        [[nodiscard]] static VulkanNativeExternalImageBacking create(
            const RuntimeExchangeEndpoint& exporter,
            const RuntimeExchangeEndpoint& consumer,
            VkExtent2D, VkFormat);
        [[nodiscard]] VulkanNativeExternalImageDescriptor exportDescriptor();
        [[nodiscard]] VkImage image() const noexcept { return imageHandle; }
        [[nodiscard]] VkDevice device() const noexcept { return endpoint.bufferDevice.device; }

    private:
        void reset() noexcept;
        RuntimeExchangeEndpoint endpoint{};
        VkImage imageHandle{};
        VkDeviceMemory memoryHandle{};
        VkDeviceSize allocationSize{};
        VkFormat format{};
        VkExtent2D extent{};
        uint32_t fourcc{};
        uint64_t modifier{};
        std::vector<VkSubresourceLayout> planes;
        bool dedicated{};
    };
}
