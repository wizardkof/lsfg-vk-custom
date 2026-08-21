/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../helpers/pointers.hpp"
#include "vulkan.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace vk {
    /// vulkan buffer
    class Buffer {
    public:
        /// create a buffer
        /// @param vk the vulkan instance
        /// @param data initial data uploaded to the buffer
        /// @param usage usage flags for the buffer
        /// @throws ls::vulkan_error on failure
        template<typename T>
        Buffer(const vk::Vulkan& vk, const T& data,
                VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
            : Buffer(vk, reinterpret_cast<const void*>(&data), sizeof(T), usage) {}

        /// create a buffer
        /// @param vk the vulkan instance
        /// @param data initial data uploaded to the buffer
        /// @param size size of the buffer in bytes
        /// @param usage usage flags for the buffer
        /// @throws ls::vulkan_error on failure
        Buffer(const vk::Vulkan& vk, const void* data, size_t size,
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        /// update an existing buffer without reallocating it
        /// @param vk the vulkan instance
        /// @param data data uploaded to the buffer
        template<typename T>
        void update(const vk::Vulkan& vk, const T& data) {
            this->update(vk, reinterpret_cast<const void*>(&data), sizeof(T));
        }

        /// update an existing buffer without reallocating it
        /// @param vk the vulkan instance
        /// @param data data uploaded to the buffer
        /// @param size size of the data in bytes
        /// @throws ls::vulkan_error on failure or if size exceeds buffer capacity
        void update(const vk::Vulkan& vk, const void* data, size_t size);

        /// Read bytes from a host-visible coherent buffer after GPU completion.
        /// @param vk the Vulkan device that owns the buffer
        /// @param size number of bytes to read
        /// @throws ls::vulkan_error if the buffer cannot be mapped or size is invalid
        [[nodiscard]] std::vector<uint8_t> read(const vk::Vulkan& vk, size_t size) const;

        /// get the buffer handle
        /// @return the buffer handle
        [[nodiscard]] const auto& handle() const { return this->buffer.get(); }
        /// get the size of the buffer
        /// @return the size of the buffer in bytes
        [[nodiscard]] size_t length() const { return this->size; }
    private:
        ls::owned_ptr<VkBuffer> buffer;
        ls::owned_ptr<VkDeviceMemory> memory;
        size_t size;
    };
}
