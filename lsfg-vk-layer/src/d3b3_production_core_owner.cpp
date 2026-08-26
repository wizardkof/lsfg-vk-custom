/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_production_core_owner.hpp"

#include <stdexcept>
#include <utility>

using namespace lsfgvk::layer;
namespace backend = lsfgvk::backend;

D3B3ProductionCoreOwner::D3B3ProductionCoreOwner(Dependencies value) :
    dependencies(std::move(value)) {
    if (!structurallyReady())
        throw std::invalid_argument("D3B3 production core owner dependencies are incomplete");
    returnSession = std::make_unique<GeneratedOutputReturnSession>(
        ProductionReturnExecution{}, *dependencies.devicePair,
        dependencies.generationEndpoint, dependencies.renderEndpoint, true);
}

D3B3ProductionCoreOwner::~D3B3ProductionCoreOwner() {
    if (generateSession && dependencies.backend)
        dependencies.backend->closeRuntimeGenerateSession(*generateSession);
}

bool D3B3ProductionCoreOwner::structurallyReady() const noexcept {
    return dependencies.backend && dependencies.devicePair && dependencies.exchangeChannel
        && dependencies.devicePair->crossDevice()
        && dependencies.extent.width != 0 && dependencies.extent.height != 0
        && dependencies.format != VK_FORMAT_UNDEFINED && dependencies.sourceImage
        && dependencies.generationEndpoint.bufferDevice.device != VK_NULL_HANDLE
        && dependencies.renderEndpoint.bufferDevice.device != VK_NULL_HANDLE;
}

backend::RuntimeGenerateSession& D3B3ProductionCoreOwner::ensureGenerateSession() {
    if (!generateSession) {
        generateSession = &dependencies.backend->openRuntimeGenerateSession(
            dependencies.extent, dependencies.format, dependencies.modifier,
            dependencies.flow, dependencies.performanceMode,
            backend::RuntimeGenerateMode::SerialReusable);
    }
    return *generateSession;
}

backend::RuntimeShadowIngestSnapshot D3B3ProductionCoreOwner::submitWarmup(
        uint64_t frameId, backend::TemporalSourceSlot slot,
        vk::RuntimeFrameTransportSubmission transport) {
    return dependencies.backend->submitRuntimeIngest(ensureGenerateSession(),
        dependencies.sourceImage(slot), std::move(transport), slot, frameId,
        backend::RuntimeIngestIntent::WARMUP_TEMPORAL);
}

backend::RuntimeIngestRetirementStatus D3B3ProductionCoreOwner::tryRetireWarmup() {
    return dependencies.backend->tryRetireRuntimeIngest(ensureGenerateSession());
}

backend::RuntimeShadowIngestSnapshot D3B3ProductionCoreOwner::submitGenerateSource(
        uint64_t frameId, backend::TemporalSourceSlot slot,
        vk::RuntimeFrameTransportSubmission transport) {
    return dependencies.backend->submitRuntimeIngest(ensureGenerateSession(),
        dependencies.sourceImage(slot), std::move(transport), slot, frameId,
        backend::RuntimeIngestIntent::GENERATE_SOURCE);
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
backend::RuntimeShadowIngestSnapshot D3B3ProductionCoreOwner::submitWarmup(
        uint64_t frameId, backend::TemporalSourceSlot slot,
        vk::SyncFdPayload payload) {
    return dependencies.backend->submitRuntimeIngest(ensureGenerateSession(),
        dependencies.sourceImage(slot), std::move(payload), slot, frameId,
        backend::RuntimeIngestIntent::WARMUP_TEMPORAL);
}

backend::RuntimeShadowIngestSnapshot D3B3ProductionCoreOwner::submitGenerateSource(
        uint64_t frameId, backend::TemporalSourceSlot slot,
        vk::SyncFdPayload payload) {
    return dependencies.backend->submitRuntimeIngest(ensureGenerateSession(),
        dependencies.sourceImage(slot), std::move(payload), slot, frameId,
        backend::RuntimeIngestIntent::GENERATE_SOURCE);
}
#endif

backend::RuntimeShadowGenerateSnapshot D3B3ProductionCoreOwner::submitGenerate(
        backend::RuntimeTemporalPairIdentity pair) {
    return dependencies.backend->submitRuntimePrepassGenerate(
        ensureGenerateSession(), pair);
}

backend::RuntimeRetirementStatus D3B3ProductionCoreOwner::tryRetireGenerate() {
    return dependencies.backend->tryRetireRuntimePrepassGenerate(ensureGenerateSession());
}

void D3B3ProductionCoreOwner::submitBReturn() {
    if (bReturnPending || returned)
        throw std::logic_error("D3B3 production return operation is already owned");
    auto pending = dependencies.backend->takeRuntimePrepassGeneratePending(
        ensureGenerateSession());
    if (!pending)
        throw std::logic_error("D3B3 production Generate has no return capability");
    bReturnPending.emplace(returnSession->submitProductionBReturn(
        std::move(*pending), *dependencies.backend, *generateSession));
}

backend::RuntimeRetirementStatus D3B3ProductionCoreOwner::tryRetireBReturn() {
    if (bReturnPending)
        return returnSession->tryRetireProductionBReturn(
            *bReturnPending, *dependencies.backend, *generateSession);
    if (!returned)
        throw std::logic_error("D3B3 production B-return operation is unavailable");
    return returnSession->tryRetireProductionBReturn(
        *returned, *dependencies.backend, *generateSession);
}

void D3B3ProductionCoreOwner::submitAReturn(vk::RuntimeForeignImageHandoffInfo handoff) {
    if (!bReturnPending || returned)
        throw std::logic_error("D3B3 production B-return is not available for A-return");
    returned.emplace(returnSession->completeProductionGeneratedReturnOnA(
        std::move(*bReturnPending), *dependencies.backend, *generateSession, handoff));
    bReturnPending.reset();
}

bool D3B3ProductionCoreOwner::tryRetireAReturn() {
    if (!returned)
        throw std::logic_error("D3B3 production A-return operation is unavailable");
    return returned->tryRetireAReturn();
}

backend::RuntimeRetirementStatus D3B3ProductionCoreOwner::releaseGeneratedOutput() {
    if (!returned)
        throw std::logic_error("D3B3 generated output operation is unavailable");
    return returnSession->releaseProductionGeneratedOutput(
        *returned, *dependencies.backend, *generateSession);
}

backend::RuntimeShadowIngestSnapshot
D3B3ProductionCoreOwner::ingestState() const noexcept {
    return generateSession
        ? dependencies.backend->inspectRuntimeIngest(*generateSession)
        : backend::RuntimeShadowIngestSnapshot{};
}

vk::RuntimeImageObservationDescriptor
D3B3ProductionCoreOwner::temporalObservationSource(
        backend::TemporalSourceSlot slot) const noexcept {
    return generateSession
        ? dependencies.backend->inspectRuntimeTemporalSource(*generateSession, slot)
        : vk::RuntimeImageObservationDescriptor{};
}

vk::RuntimeImageObservationDescriptor
D3B3ProductionCoreOwner::generatedObservationSource() const noexcept {
    return generateSession
        ? dependencies.backend->inspectRuntimeGeneratedOutput(*generateSession)
        : vk::RuntimeImageObservationDescriptor{};
}

backend::RuntimeShadowGenerateSnapshot
D3B3ProductionCoreOwner::generateState() const noexcept {
    return generateSession
        ? dependencies.backend->inspectRuntimePrepassGenerate(*generateSession)
        : backend::RuntimeShadowGenerateSnapshot{};
}

const backend::ReturnedGeneratedOperation*
D3B3ProductionCoreOwner::returnedOperation() const noexcept {
    return returned ? &*returned : nullptr;
}

backend::ReturnedGeneratedOperation
D3B3ProductionCoreOwner::takeReturnedOperation() {
    if (!returned || !returned->terminalReady())
        throw std::logic_error("D3B3 returned operation is not terminal-ready");
    auto value = std::move(*returned);
    returned.reset();
    returnSession->resetAfterProductionHandoff();
    return value;
}
