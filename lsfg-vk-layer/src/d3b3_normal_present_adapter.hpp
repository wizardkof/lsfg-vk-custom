/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "d3b3_production_seams.hpp"

#include <cstdint>
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

namespace lsfgvk::layer {

class D3B3AsyncFiniteComposition;

enum class D3B3ProductionInputDomain : uint8_t {
    FRAME_TRANSPORT,
    FOREIGN_HANDOFF,
    ORIGINAL_READY,
    TERMINAL_D3B2,
    WARMUP_ORIGINAL
};

/// Nonblocking accounting for provider-owned production resources.  Tokens
/// carry no raw Vulkan handles: a future production assembly keeps the real
/// move-only authority and gives the adapter only this scoped lifetime proof.
class D3B3ProductionResourceState {
public:
    class Token {
    public:
        Token() noexcept = default;
        Token(const Token&) = delete;
        Token& operator=(const Token&) = delete;
        Token(Token&&) noexcept;
        Token& operator=(Token&&) noexcept;
        ~Token();
        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] D3B3ProductionInputDomain domain() const noexcept { return kind; }
        [[nodiscard]] uintptr_t scope() const noexcept { return scopeValue; }
        [[nodiscard]] uint64_t generation() const noexcept { return generationValue; }
        [[nodiscard]] uint64_t epoch() const noexcept { return epochValue; }
        void retire() noexcept;
    private:
        friend class D3B3ProductionResourceState;
        Token(D3B3ProductionResourceState*, D3B3ProductionInputDomain,
            uintptr_t, uint64_t, uint64_t) noexcept;
        D3B3ProductionResourceState* owner{};
        D3B3ProductionInputDomain kind{D3B3ProductionInputDomain::FRAME_TRANSPORT};
        uintptr_t scopeValue{};
        uint64_t generationValue{};
        uint64_t epochValue{};
    };

    [[nodiscard]] Token acquire(D3B3ProductionInputDomain, uintptr_t scope,
        uint64_t generation, uint64_t epoch) noexcept;
    [[nodiscard]] bool retirementReady() const noexcept {
        return outstanding.load(std::memory_order_acquire) == 0;
    }
    [[nodiscard]] uint32_t outstandingAuthorities() const noexcept {
        return outstanding.load(std::memory_order_acquire);
    }
private:
    std::atomic<uint32_t> outstanding{};
};

struct D3B3AsyncPresentInput {
    D3B3AsyncPresentInput() noexcept = default;
    D3B3AsyncPresentInput(VkSwapchainKHR value, uint64_t generation,
        uint64_t frame) noexcept : swapchain(value),
        swapchainGeneration(generation), frameId(frame) {}
    D3B3AsyncPresentInput(const D3B3AsyncPresentInput&) = delete;
    D3B3AsyncPresentInput& operator=(const D3B3AsyncPresentInput&) = delete;
    D3B3AsyncPresentInput(D3B3AsyncPresentInput&&) noexcept = default;
    D3B3AsyncPresentInput& operator=(D3B3AsyncPresentInput&&) noexcept = default;
    VkSwapchainKHR swapchain{};
    uint64_t swapchainGeneration{};
    uint64_t frameId{};
    std::vector<D3B3ProductionResourceState::Token> authorities;
};

enum class D3B3NormalAdapterResult : uint8_t {
    NOT_SELECTED,
    READY,
    TEMPORARILY_BLOCKED,
    FAILED_BEFORE_OWNERSHIP,
    FAILED_AFTER_OWNERSHIP,
    DEVICE_LOST,
    FINITE_COMPLETE
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
        D3B3AsyncFiniteComposition& composition);

    [[nodiscard]] bool structurallyReady() const noexcept;
    [[nodiscard]] D3B3NormalAdapterResult readiness() const noexcept;
    [[nodiscard]] D3B3PresentOwnership ownership() const noexcept { return ownershipState; }
    [[nodiscard]] uint64_t frameSerial() const noexcept { return nextFrameSerial; }
    [[nodiscard]] uint64_t swapchainGeneration() const noexcept {
        return contextValue.swapchainGeneration;
    }
    [[nodiscard]] const D3B3NormalPresentContext& context() const noexcept { return contextValue; }
    [[nodiscard]] D3B3AsyncFiniteComposition* composition() noexcept {
        return asyncComposition;
    }

    /// Construct the controller's operation seam without starting a frame.
    /// This is the only adapter operation factory boundary; no Vulkan work is
    /// performed and no normal present is consumed here.
    bool constructOperations();
    D3B3NormalAdapterResult processFrame(uint64_t frameId) noexcept;
    D3B3NormalAdapterResult processFrame(D3B3AsyncPresentInput&&) noexcept;

    /// Records a future ownership transfer. The live route does not call this
    /// until a later cutover stage explicitly selects the adapter.
    D3B3NormalAdapterResult acceptOwnership() noexcept;
    void resetForSwapchainGeneration(D3B3NormalPresentContext replacement);

private:
    D3B3NormalPresentContext contextValue;
    D3B3AsyncFiniteComposition* asyncComposition{};
    D3B3PresentOwnership ownershipState{D3B3PresentOwnership::OUTER_ROUTE};
    D3B3NormalAdapterResult currentResult{D3B3NormalAdapterResult::NOT_SELECTED};
    uint64_t nextFrameSerial{1};
    std::optional<D3B3AsyncPresentInput> activeInput;
};

}
