/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b3_production_runtime.hpp"

#include <cstdint>
#include <memory>

namespace lsfgvk::layer {

enum class D3B3NormalAdapterResult : uint8_t {
    NOT_SELECTED,
    READY,
    TEMPORARILY_BLOCKED,
    FAILED_BEFORE_OWNERSHIP,
    FAILED_AFTER_OWNERSHIP,
    DEVICE_LOST
};

enum class D3B3PresentOwnership : uint8_t {
    OUTER_ROUTE,
    D3B3_ROUTE,
    TERMINAL
};

/// Immutable values available at the normal present boundary. Handles are
/// observed only; the adapter does not acquire, submit, or present anything.
struct D3B3NormalPresentContext {
    VkSwapchainKHR swapchain{};
    VkImage sourceImage{};
    VkImageLayout sourceLayout{VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkExtent2D extent{};
    VkQueue presentQueue{};
    uint32_t presentQueueFamily{VK_QUEUE_FAMILY_IGNORED};
    VkSemaphore originalReady{};
    uint64_t swapchainGeneration{};
    bool runtimeDevicePairReady{};
    bool exchangeChannelReady{};
    bool terminalReady{};
};

/// Structural production seam for the future finite normal-route cutover.
/// It owns one controller per swapchain and deliberately has no live-route
/// entry point from Swapchain::present() at this stage.
class D3B3NormalPresentAdapter {
public:
    explicit D3B3NormalPresentAdapter(D3B3NormalPresentContext context,
        std::unique_ptr<D3B3ProductionRuntimeSession> runtime = {});

    [[nodiscard]] bool structurallyReady() const noexcept;
    [[nodiscard]] D3B3NormalAdapterResult readiness() const noexcept;
    [[nodiscard]] D3B3PresentOwnership ownership() const noexcept { return ownershipState; }
    [[nodiscard]] uint64_t frameSerial() const noexcept { return nextFrameSerial; }
    [[nodiscard]] uint64_t swapchainGeneration() const noexcept {
        return contextValue.swapchainGeneration;
    }
    [[nodiscard]] const D3B3NormalPresentContext& context() const noexcept { return contextValue; }
    [[nodiscard]] D3B3ProductionState* controller() noexcept {
        return runtimeSession ? &runtimeSession->controller() : nullptr;
    }

    /// Construct the controller's operation seam without starting a frame.
    /// This is the only adapter operation factory boundary; no Vulkan work is
    /// performed and no normal present is consumed here.
    bool constructOperations();
    D3B3NormalAdapterResult processFrame(uint64_t frameId) noexcept;

    /// Records a future ownership transfer. The live route does not call this
    /// until a later cutover stage explicitly selects the adapter.
    D3B3NormalAdapterResult acceptOwnership() noexcept;
    void resetForSwapchainGeneration(D3B3NormalPresentContext replacement);

private:
    D3B3NormalPresentContext contextValue;
    std::unique_ptr<D3B3ProductionRuntimeSession> runtimeSession;
    D3B3PresentOwnership ownershipState{D3B3PresentOwnership::OUTER_ROUTE};
    D3B3NormalAdapterResult currentResult{D3B3NormalAdapterResult::NOT_SELECTED};
    uint64_t nextFrameSerial{1};
};

}
