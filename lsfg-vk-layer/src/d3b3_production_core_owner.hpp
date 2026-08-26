/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "generated_output_return_diagnostic.hpp"

#include <functional>
#include <optional>

namespace lsfgvk::layer {

/// Minimal production-compiled owner for the already-qualified D3B3 GPU
/// cores.  It deliberately owns no WSI, present, terminal, or routing policy.
class D3B3ProductionCoreOwner {
public:
    struct Dependencies {
        backend::Instance* backend{};
        const vk::RuntimeDevicePair* devicePair{};
        const vk::RuntimeExchangeChannel* exchangeChannel{};
        vk::RuntimeExchangeEndpoint generationEndpoint{};
        vk::RuntimeExchangeEndpoint renderEndpoint{};
        VkExtent2D extent{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        uint64_t modifier{};
        float flow{};
        bool performanceMode{};
        std::function<VkImage(backend::TemporalSourceSlot)> sourceImage;
    };

    explicit D3B3ProductionCoreOwner(Dependencies);
    D3B3ProductionCoreOwner(const D3B3ProductionCoreOwner&) = delete;
    D3B3ProductionCoreOwner& operator=(const D3B3ProductionCoreOwner&) = delete;
    ~D3B3ProductionCoreOwner();

    [[nodiscard]] bool structurallyReady() const noexcept;
    [[nodiscard]] backend::RuntimeShadowIngestSnapshot submitWarmup(
        uint64_t frameId, backend::TemporalSourceSlot,
        vk::RuntimeFrameTransportSubmission);
    [[nodiscard]] backend::RuntimeIngestRetirementStatus tryRetireWarmup();
    [[nodiscard]] backend::RuntimeShadowIngestSnapshot submitGenerateSource(
        uint64_t frameId, backend::TemporalSourceSlot,
        vk::RuntimeFrameTransportSubmission);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    [[nodiscard]] backend::RuntimeShadowIngestSnapshot submitWarmup(
        uint64_t frameId, backend::TemporalSourceSlot, vk::SyncFdPayload);
    [[nodiscard]] backend::RuntimeShadowIngestSnapshot submitGenerateSource(
        uint64_t frameId, backend::TemporalSourceSlot, vk::SyncFdPayload);
#endif
    [[nodiscard]] backend::RuntimeShadowGenerateSnapshot submitGenerate(
        backend::RuntimeTemporalPairIdentity);
    [[nodiscard]] backend::RuntimeRetirementStatus tryRetireGenerate();
    void submitBReturn();
    [[nodiscard]] backend::RuntimeRetirementStatus tryRetireBReturn();
    void submitAReturn(vk::RuntimeForeignImageHandoffInfo);
    [[nodiscard]] bool tryRetireAReturn();
    [[nodiscard]] backend::RuntimeRetirementStatus releaseGeneratedOutput();

    [[nodiscard]] backend::RuntimeShadowIngestSnapshot ingestState() const noexcept;
    [[nodiscard]] backend::RuntimeShadowGenerateSnapshot generateState() const noexcept;
    [[nodiscard]] vk::RuntimeImageObservationDescriptor temporalObservationSource(
        backend::TemporalSourceSlot) const noexcept;
    [[nodiscard]] vk::RuntimeImageObservationDescriptor
        generatedObservationSource() const noexcept;
    [[nodiscard]] const backend::ReturnedGeneratedOperation* returnedOperation() const noexcept;
    [[nodiscard]] backend::ReturnedGeneratedOperation takeReturnedOperation();

private:
    backend::RuntimeGenerateSession& ensureGenerateSession();

    Dependencies dependencies;
    backend::RuntimeGenerateSession* generateSession{};
    std::unique_ptr<GeneratedOutputReturnSession> returnSession;
    std::optional<RuntimeGeneratedBReturnPending> bReturnPending;
    std::optional<backend::ReturnedGeneratedOperation> returned;
};

}
