#pragma once

#include <atomic>
#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace lsfgvk::layer {
    enum class GraphicsFinalExecutionMode : uint8_t {
        INELIGIBLE,
        DEDICATED_ASYNC,
        BORROWED_SYNCHRONOUS
    };

    struct GraphicsFinalQueueInfo {
        VkQueue queue{VK_NULL_HANDLE};
        uint32_t family{VK_QUEUE_FAMILY_IGNORED};
        uint32_t index{};
        VkQueueFlags flags{};
        bool surfacePresentSupported{};
        bool borrowed{};
    };

    [[nodiscard]] constexpr GraphicsFinalExecutionMode graphicsFinalExecutionMode(
            const GraphicsFinalQueueInfo& info, bool synchronousOperation) noexcept {
        if (info.queue == VK_NULL_HANDLE
                || info.family == VK_QUEUE_FAMILY_IGNORED
                || !(info.flags & VK_QUEUE_GRAPHICS_BIT)
                || !info.surfacePresentSupported)
            return GraphicsFinalExecutionMode::INELIGIBLE;
        if (info.borrowed)
            return synchronousOperation
                ? GraphicsFinalExecutionMode::BORROWED_SYNCHRONOUS
                : GraphicsFinalExecutionMode::INELIGIBLE;
        return GraphicsFinalExecutionMode::DEDICATED_ASYNC;
    }

    class BorrowedGraphicsQueueLease {
    public:
        enum class State : uint8_t { ACTIVE, RELEASED };

        BorrowedGraphicsQueueLease(VkQueue queue, uint32_t family,
                uint32_t index, uint64_t operation) noexcept :
            queueHandle(queue), familyIndex(family), queueIndex(index),
            operationIdentity(operation) {}

        [[nodiscard]] bool validFor(VkQueue queue, uint32_t family) const noexcept {
            return this->state.load() == State::ACTIVE
                && queue == this->queueHandle && family == this->familyIndex;
        }
        [[nodiscard]] bool release() noexcept {
            auto expected = State::ACTIVE;
            return this->state.compare_exchange_strong(expected, State::RELEASED);
        }
        [[nodiscard]] State currentState() const noexcept { return this->state.load(); }
        [[nodiscard]] VkQueue queue() const noexcept { return this->queueHandle; }
        [[nodiscard]] uint32_t family() const noexcept { return this->familyIndex; }
        [[nodiscard]] uint32_t index() const noexcept { return this->queueIndex; }
        [[nodiscard]] uint64_t operation() const noexcept { return this->operationIdentity; }

    private:
        VkQueue queueHandle{VK_NULL_HANDLE};
        uint32_t familyIndex{VK_QUEUE_FAMILY_IGNORED};
        uint32_t queueIndex{};
        uint64_t operationIdentity{};
        std::atomic<State> state{State::ACTIVE};
    };
}
