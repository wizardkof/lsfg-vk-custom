/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_per_swapchain_runtime.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <stdexcept>
#include <utility>

namespace lsfgvk::layer {

namespace {
bool complete(const D3B3AsyncFiniteOperations& value) {
    return value.submitIngest && value.tryRetireIngest
        && value.presentWarmupOriginal && value.submitGenerate
        && value.tryRetireGenerate
        && value.submitBReturn && value.tryRetireBReturn
        && value.submitAReturn && value.releaseGeneratedOutput
        && value.takeReturnedOperation && value.makeOriginal && value.makeTerminal;
}
}

D3B3AsyncFiniteComposition::D3B3AsyncFiniteComposition(
        D3B3AsyncFiniteOperations value) : operations(std::move(value)) {
    if (!complete(operations))
        throw std::invalid_argument("incomplete asynchronous D3B3 finite operations");
}

bool D3B3AsyncFiniteComposition::retirementReady() const noexcept {
    return current == D3B3PrePairPhase::IDLE && !pair && activeFrame == 0;
}

D3B3AsyncProgress D3B3AsyncFiniteComposition::processFrame(uint64_t frameId) noexcept {
    if (current == D3B3PrePairPhase::FAILED)
        return D3B3AsyncProgress::FAILED;
    if (totals.frames == 5 && retirementReady())
        return D3B3AsyncProgress::FINISHED;
    if (frameId == 0 || totals.frames >= 5
            || (activeFrame != 0 && activeFrame != frameId))
        return D3B3AsyncProgress::FAILED;
    try {
        return activeFrame == 0 ? beginFrame(frameId) : advanceActive();
    } catch (const ls::vulkan_error& error) {
        fail();
        return error.error() == VK_ERROR_DEVICE_LOST
            ? D3B3AsyncProgress::DEVICE_LOST : D3B3AsyncProgress::FAILED;
    } catch (...) {
        fail();
        return D3B3AsyncProgress::FAILED;
    }
}

D3B3AsyncProgress D3B3AsyncFiniteComposition::beginFrame(uint64_t frameId) {
    if (current != D3B3PrePairPhase::IDLE)
        throw std::logic_error("D3B3 finite frame began with active phase");
    const auto frameNumber = totals.frames + 1;
    const auto slot = frameNumber == 1 || frameNumber == 3 || frameNumber == 5
        ? backend::TemporalSourceSlot::Slot0 : backend::TemporalSourceSlot::Slot1;
    const uint64_t ingestEpoch = operations.submitIngest(
        frameId, slot, frameNumber == 1);
    if (ingestEpoch == 0)
        throw std::runtime_error("D3B3 ingest submit rejected");
    activeFrame = frameId;
    if (frameNumber <= 2) {
        current = D3B3PrePairPhase::WARMUP_SUBMITTED;
        return D3B3AsyncProgress::PENDING;
    }

    const size_t newerIndex = backend::temporalSourceSlotIndex(slot);
    const size_t olderIndex = 1U - newerIndex;
    if (slots[olderIndex].state != D3B3TemporalState::RETAINED_HISTORY)
        throw std::logic_error("D3B3 asynchronous temporal history is unavailable");
    activeIdentity = {
        .olderSlot = olderIndex == 0 ? backend::TemporalSourceSlot::Slot0
                                    : backend::TemporalSourceSlot::Slot1,
        .newerSlot = slot,
        .olderFrameId = slots[olderIndex].frameId,
        .newerFrameId = frameId,
        .generationId = ingestEpoch};
    if (activeIdentity.generationId <= generation)
        throw std::logic_error("D3B3 ingest epoch did not advance");
    generation = activeIdentity.generationId;
    if (!operations.submitGenerate(activeIdentity))
        throw std::runtime_error("D3B3 Generate submit rejected");
    ++totals.generateSubmits;
    current = D3B3PrePairPhase::GENERATE_SUBMITTED;
    operations.submitBReturn();
    ++totals.bReturnSubmits;
    current = D3B3PrePairPhase::B_RETURN_SUBMITTED;
    operations.submitAReturn(activeIdentity);
    ++totals.aReturnSubmits;
    current = D3B3PrePairPhase::A_RETURN_SUBMITTED;
    return D3B3AsyncProgress::PENDING;
}

D3B3AsyncProgress D3B3AsyncFiniteComposition::advanceActive() {
    if (current == D3B3PrePairPhase::WARMUP_SUBMITTED) {
        const auto status = operations.tryRetireIngest();
        if (status == D3B3RetirementStatus::TIMEOUT) {
            ++totals.pendingPolls;
            return D3B3AsyncProgress::PENDING;
        }
        if (status == D3B3RetirementStatus::DEVICE_LOST) {
            fail();
            return D3B3AsyncProgress::DEVICE_LOST;
        }
        if (status != D3B3RetirementStatus::RETIRED
                || !operations.presentWarmupOriginal(activeFrame))
            throw std::runtime_error("D3B3 warmup retirement/present failed");
        const size_t index = totals.frames == 0 ? 0 : 1;
        slots[index] = {D3B3TemporalState::RETAINED_HISTORY, activeFrame, 0};
        if (totals.frames == 1)
            slots[0].state = D3B3TemporalState::OVERWRITEABLE;
        ++totals.frames;
        activeFrame = 0;
        current = D3B3PrePairPhase::IDLE;
        return D3B3AsyncProgress::ACCEPTED;
    }

    if (current == D3B3PrePairPhase::A_RETURN_SUBMITTED) {
        if (!generateRetired) {
            const auto status = operations.tryRetireGenerate();
            if (status == backend::RuntimeRetirementStatus::DEVICE_LOST) {
                fail();
                return D3B3AsyncProgress::DEVICE_LOST;
            }
            if (status == backend::RuntimeRetirementStatus::FAILED)
                throw std::runtime_error("D3B3 Generate retirement failed");
            generateRetired = status == backend::RuntimeRetirementStatus::RETIRED;
        }
        if (!bReturnRetired) {
            const auto status = operations.tryRetireBReturn();
            if (status == backend::RuntimeRetirementStatus::DEVICE_LOST) {
                fail();
                return D3B3AsyncProgress::DEVICE_LOST;
            }
            if (status == backend::RuntimeRetirementStatus::FAILED)
                throw std::runtime_error("D3B3 B-return retirement failed");
            bReturnRetired = status == backend::RuntimeRetirementStatus::RETIRED;
        }
        if (!generateRetired || !bReturnRetired) {
            ++totals.pendingPolls;
            return D3B3AsyncProgress::PENDING;
        }
        if (!generatedOutputReleased) {
            const auto status = operations.releaseGeneratedOutput();
            if (status == backend::RuntimeRetirementStatus::DEVICE_LOST) {
                fail();
                return D3B3AsyncProgress::DEVICE_LOST;
            }
            if (status != backend::RuntimeRetirementStatus::RETIRED)
                throw std::runtime_error("D3B3 generated output release failed");
            generatedOutputReleased = true;
        }
        auto original = operations.makeOriginal(activeIdentity);
        auto terminal = operations.makeTerminal(activeIdentity);
        if (!original.valid()
                || original.source().frameId != activeIdentity.newerFrameId
                || !terminal.originalReadyAuthority
                || !terminal.originalReadyAuthority->valid()
                || terminal.originalReadyAuthority->epoch()
                    != activeIdentity.generationId
                || terminal.originalReadyAuthority->scope()
                    != original.source().authorityScope
                || !terminal.build || !terminal.tryRetireGeneratedPresentFence
                || !terminal.tryRetireOriginalPresentFence)
            throw std::runtime_error("D3B3 terminal inputs are invalid");
        auto returned = operations.takeReturnedOperation();
        if (!returned.terminalReady() || returned.identity().generationId
                != activeIdentity.generationId)
            throw std::runtime_error("D3B3 terminal handoff authority is invalid");
        current = D3B3PrePairPhase::TERMINAL_READY;
        pair.emplace(std::move(returned), std::move(original), std::move(terminal));
        ++totals.pairConstructions;
        pair->preflight();
        pair->submitTerminal();
        ++totals.terminalSubmits;
        current = D3B3PrePairPhase::PAIR_SUBMITTED;
        terminalRetired = pair->state() == D3B3PairState::GRAPHICS_RETIRED;
        aReturnRetired = pair->aReturnRetired();
        return D3B3AsyncProgress::PENDING;
    }

    if (current != D3B3PrePairPhase::PAIR_SUBMITTED || !pair)
        throw std::logic_error("D3B3 asynchronous phase cannot advance");

    if (!terminalRetired) {
        const auto result = pair->tryRetireTerminal();
        if (result == D3B2RetirementResult::DEVICE_LOST) {
            fail();
            return D3B3AsyncProgress::DEVICE_LOST;
        }
        if (result == D3B2RetirementResult::FAILED)
            throw std::runtime_error("D3B3 terminal retirement failed");
        terminalRetired = result == D3B2RetirementResult::RETIRED;
    }
    if (terminalRetired && pair->state() == D3B3PairState::GRAPHICS_RETIRED) {
        const auto result = pair->tryRetirePresentWaits();
        if (result == D3B2RetirementResult::DEVICE_LOST) {
            fail();
            return D3B3AsyncProgress::DEVICE_LOST;
        }
        if (result == D3B2RetirementResult::FAILED)
            throw std::runtime_error("D3B3 present retirement failed");
    }
    if (!aReturnRetired)
        aReturnRetired = pair->tryRetireAReturn();
    if (!terminalRetired || !aReturnRetired
            || pair->state() != D3B3PairState::PRESENTS_RETIRED) {
        ++totals.pendingPolls;
        return D3B3AsyncProgress::PENDING;
    }
    pair->retirePair();
    ++totals.pairRetirements;
    current = D3B3PrePairPhase::PAIR_RETIRED;
    const size_t newer = backend::temporalSourceSlotIndex(activeIdentity.newerSlot);
    const size_t older = backend::temporalSourceSlotIndex(activeIdentity.olderSlot);
    slots[older].state = D3B3TemporalState::OVERWRITEABLE;
    slots[newer] = {D3B3TemporalState::RETAINED_HISTORY,
        activeIdentity.newerFrameId, activeIdentity.generationId};
    ++totals.frames;
    pair.reset();
    activeIdentity = {};
    activeFrame = 0;
    terminalRetired = false;
    aReturnRetired = false;
    generateRetired = false;
    bReturnRetired = false;
    generatedOutputReleased = false;
    current = D3B3PrePairPhase::IDLE;
    return totals.frames == 5
        ? D3B3AsyncProgress::FINISHED : D3B3AsyncProgress::ACCEPTED;
}

D3B3PerSwapchainRuntimeOwner::D3B3PerSwapchainRuntimeOwner(
        D3B3PerSwapchainRuntimeDescriptor descriptor,
        const D3B3PerSwapchainRuntimeAssembly& assembly) {
    if (descriptor.present.swapchain == VK_NULL_HANDLE
            || descriptor.present.swapchainGeneration == 0
            || !assembly.bindFiniteOperations || !assembly.bindRetirementReady
            || !assembly.bindAsyncFiniteOperations)
        throw std::invalid_argument("incomplete D3B3 per-swapchain assembly");

    auto localCore = std::make_unique<D3B3ProductionCoreOwner>(descriptor.core);
    auto retirement = assembly.bindRetirementReady(*localCore);
    if (!retirement)
        throw std::invalid_argument("D3B3 retirement authority is incomplete");
    auto operations = assembly.bindFiniteOperations(*localCore, descriptor);
    auto asyncFinite = std::make_unique<D3B3AsyncFiniteComposition>(
        assembly.bindAsyncFiniteOperations(*localCore, descriptor));
    auto runtime = std::make_unique<D3B3ProductionRuntimeSession>(
        descriptor.present.swapchainGeneration, std::move(operations));
    if (!runtime->structurallyReady())
        throw std::invalid_argument("D3B3 finite runtime assembly is incomplete");
    auto adapter = std::make_unique<D3B3NormalPresentAdapter>(
        descriptor.present, std::move(runtime));
    if (!adapter->constructOperations())
        throw std::invalid_argument("D3B3 normal adapter prerequisites are incomplete");

    core = std::move(localCore);
    isRetirementReady = std::move(retirement);
    this->asyncFinite = std::move(asyncFinite);
    normalAdapter = std::move(adapter);
}

VkSwapchainKHR D3B3PerSwapchainRuntimeOwner::swapchain() const noexcept {
    return normalAdapter ? normalAdapter->context().swapchain : VK_NULL_HANDLE;
}

uint64_t D3B3PerSwapchainRuntimeOwner::generation() const noexcept {
    return normalAdapter ? normalAdapter->swapchainGeneration() : 0;
}

bool D3B3PerSwapchainRuntimeOwner::structurallyReady() const noexcept {
    return core && core->structurallyReady() && normalAdapter
        && normalAdapter->structurallyReady();
}

bool D3B3PerSwapchainRuntimeOwner::retirementReady() const noexcept {
    try {
        return asyncFinite && asyncFinite->retirementReady()
            && isRetirementReady && isRetirementReady();
    } catch (...) {
        return false;
    }
}

std::shared_ptr<D3B3PerSwapchainRuntimeFactory::Owner>
D3B3PerSwapchainRuntimeFactory::create(
        D3B3PerSwapchainRuntimeDescriptor descriptor,
        const D3B3PerSwapchainRuntimeAssembly& assembly) {
    auto candidate = std::make_shared<Owner>(std::move(descriptor), assembly);
    const auto key = candidate->swapchain();
    std::scoped_lock lock(mutex);
    const auto [iterator, inserted] = owners.emplace(key, candidate);
    if (!inserted)
        throw std::logic_error("D3B3 swapchain runtime already exists");
    return iterator->second;
}

std::shared_ptr<D3B3PerSwapchainRuntimeFactory::Owner>
D3B3PerSwapchainRuntimeFactory::recreate(
        D3B3PerSwapchainRuntimeDescriptor descriptor,
        const D3B3PerSwapchainRuntimeAssembly& assembly) {
    auto candidate = std::make_shared<Owner>(std::move(descriptor), assembly);
    const auto key = candidate->swapchain();
    std::scoped_lock lock(mutex);
    auto iterator = owners.find(key);
    if (iterator == owners.end())
        throw std::logic_error("D3B3 swapchain runtime does not exist");
    if (candidate->generation() <= iterator->second->generation())
        throw std::invalid_argument("D3B3 swapchain generation did not advance");
    if (!iterator->second->retirementReady())
        throw std::logic_error("D3B3 previous swapchain generation is still active");
    iterator->second = candidate;
    return candidate;
}

std::shared_ptr<D3B3PerSwapchainRuntimeFactory::Owner>
D3B3PerSwapchainRuntimeFactory::find(VkSwapchainKHR swapchain) const {
    std::scoped_lock lock(mutex);
    const auto iterator = owners.find(swapchain);
    return iterator == owners.end() ? nullptr : iterator->second;
}

bool D3B3PerSwapchainRuntimeFactory::destroy(VkSwapchainKHR swapchain) {
    std::scoped_lock lock(mutex);
    const auto iterator = owners.find(swapchain);
    if (iterator == owners.end() || !iterator->second->retirementReady())
        return false;
    owners.erase(iterator);
    return true;
}

size_t D3B3PerSwapchainRuntimeFactory::size() const noexcept {
    std::scoped_lock lock(mutex);
    return owners.size();
}

}
