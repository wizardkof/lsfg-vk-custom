#include "generated_output_return_diagnostic.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/fnv1a.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

using namespace lsfgvk::layer;

namespace {
struct ShadowBReturnResourceLifetime {
    vk::RuntimeExchangeEndpoint endpoint;
    vk::VulkanNativeExternalImageBacking backing;
    vk::SyncFdSemaphore signal;
    VkCommandPool commandPool{};
    VkCommandBuffer command{};

    ~ShadowBReturnResourceLifetime() {
        if (command && commandPool && endpoint.FreeCommandBuffers)
            endpoint.FreeCommandBuffers(endpoint.bufferDevice.device,
                commandPool, 1, &command);
        if (commandPool && endpoint.DestroyCommandPool)
            endpoint.DestroyCommandPool(endpoint.bufferDevice.device,
                commandPool, nullptr);
    }
};
}

bool lsfgvk::layer::generatedOutputIntegrityMatches(
        const GeneratedOutputIntegrity& a, const GeneratedOutputIntegrity& b) noexcept {
    return a.generation != 0 && a.generation == b.generation
        && a.extent.width == b.extent.width && a.extent.height == b.extent.height
        && a.format == b.format && a.byteCount == b.byteCount
        && a.nonzeroByteCount == b.nonzeroByteCount && a.checksum == b.checksum
        && a.byteCount == static_cast<size_t>(a.extent.width) * a.extent.height * 4U
        && a.nonzeroByteCount > 0;
}

void GeneratedOutputReturnStateMachine::advance(
        GeneratedOutputReturnState expected, GeneratedOutputReturnState next) {
    if (current != expected || current == GeneratedOutputReturnState::FAILED
            || current == GeneratedOutputReturnState::PASS
            || static_cast<int>(next) != static_cast<int>(expected) + 1)
        throw std::logic_error("invalid D3A1 state transition");
    current = next;
}

void GpuChainedReturnStateMachine::advance(
        GpuChainedReturnState expected, GpuChainedReturnState next) {
    if (current != expected || current == GpuChainedReturnState::FAILED
            || current == GpuChainedReturnState::PASS
            || static_cast<int>(next) != static_cast<int>(expected) + 1)
        throw std::logic_error("invalid D3A2 state transition");
    current = next;
}

void GpuChainedReturnStateMachine::resetAfterProductionHandoff() {
    if (current != GpuChainedReturnState::A_SUBMITTED)
        throw std::logic_error("D3A2 production handoff is not resettable");
    current = GpuChainedReturnState::EMPTY;
}

GeneratedOutputReturnSession::GeneratedOutputReturnSession(
        backend::RuntimeGenerateDiagnosticResult&& result, const vk::RuntimeDevicePair& pair,
        vk::RuntimeExchangeEndpoint generation, vk::RuntimeExchangeEndpoint render,
        bool captureOnly) : generationEndpoint(std::move(generation)),
    renderEndpoint(std::move(render)) {
    if (!generatedOutputRoleBindingEligible(pair, generationEndpoint.identity,
            renderEndpoint.identity, captureOnly))
        throw std::invalid_argument("D3A1 requires CROSS_PHYSICAL_DEVICE + CAPTURE_ONLY");
    if (generationEndpoint.bufferDevice.device == renderEndpoint.bufferDevice.device)
        throw std::invalid_argument("D3A1 endpoint role mismatch");
    execute(std::move(result));
}

GeneratedOutputReturnSession::GeneratedOutputReturnSession(
        backend::RuntimeGenerateDiagnosticPending&& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& session,
        const vk::RuntimeDevicePair& pair, vk::RuntimeExchangeEndpoint generation,
        vk::RuntimeExchangeEndpoint render, bool captureOnly,
        std::optional<vk::RuntimeForeignImageHandoffInfo> handoff) :
    generationEndpoint(std::move(generation)), renderEndpoint(std::move(render)) {
    if (!generatedOutputRoleBindingEligible(pair, generationEndpoint.identity,
            renderEndpoint.identity, captureOnly))
        throw std::invalid_argument("D3A2 requires CROSS_PHYSICAL_DEVICE + CAPTURE_ONLY");
    if (generationEndpoint.bufferDevice.device == renderEndpoint.bufferDevice.device)
        throw std::invalid_argument("D3A2 endpoint role mismatch");
    executeGpuChained(std::move(pending), backend, session, handoff);
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
GeneratedOutputReturnSession::GeneratedOutputReturnSession(
        ShadowReturnExecutionForTesting, const vk::RuntimeDevicePair& pair,
        vk::RuntimeExchangeEndpoint generation, vk::RuntimeExchangeEndpoint render,
        bool captureOnly) : generationEndpoint(std::move(generation)),
    renderEndpoint(std::move(render)) {
    if (!generatedOutputRoleBindingEligible(pair, generationEndpoint.identity,
            renderEndpoint.identity, captureOnly))
        throw std::invalid_argument("shadow return requires CROSS_PHYSICAL_DEVICE + CAPTURE_ONLY");
    if (generationEndpoint.bufferDevice.device == renderEndpoint.bufferDevice.device)
        throw std::invalid_argument("shadow return endpoint role mismatch");
}
#endif

GeneratedOutputReturnSession::GeneratedOutputReturnSession(
        ProductionReturnExecution, const vk::RuntimeDevicePair& pair,
        vk::RuntimeExchangeEndpoint generation, vk::RuntimeExchangeEndpoint render,
        bool captureOnly) : generationEndpoint(std::move(generation)),
    renderEndpoint(std::move(render)) {
    if (!generatedOutputRoleBindingEligible(pair, generationEndpoint.identity,
            renderEndpoint.identity, captureOnly))
        throw std::invalid_argument("production return requires CROSS_PHYSICAL_DEVICE + CAPTURE_ONLY");
    if (generationEndpoint.bufferDevice.device == renderEndpoint.bufferDevice.device)
        throw std::invalid_argument("production return endpoint role mismatch");
}

GeneratedOutputReturnSession::~GeneratedOutputReturnSession() {
    resetBCommands();
}

void GeneratedOutputReturnSession::resetBCommands() noexcept {
    if (bCommand && bCommandPool && generationEndpoint.FreeCommandBuffers)
        generationEndpoint.FreeCommandBuffers(generationEndpoint.bufferDevice.device,
            bCommandPool, 1, &bCommand);
    if (bCommandPool && generationEndpoint.DestroyCommandPool)
        generationEndpoint.DestroyCommandPool(generationEndpoint.bufferDevice.device,
            bCommandPool, nullptr);
    bCommand = VK_NULL_HANDLE; bCommandPool = VK_NULL_HANDLE;
}

void GeneratedOutputReturnSession::advance(
        GeneratedOutputReturnState expected, GeneratedOutputReturnState next) {
    states.advance(expected, next);
}

void GeneratedOutputReturnSession::execute(
        backend::RuntimeGenerateDiagnosticResult&& result) {
    try {
        if (!result.frame.valid() || result.metadata.generation != result.frame.identity()
                || result.metadata.extent.width != result.frame.extentValue().width
                || result.metadata.extent.height != result.frame.extentValue().height
                || result.metadata.format != result.frame.formatValue()
                || result.frame.layoutValue() != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                || result.frame.queueFamily() != generationEndpoint.queueFamilyIndex)
            throw std::invalid_argument("invalid or stale D3A1 generated-frame capability");
        authoritative = {result.metadata.generation, result.metadata.extent,
            result.metadata.format, result.metadata.byteCount,
            result.metadata.nonzeroByteCount, result.metadata.checksum};
        const VkImage sourceImage = result.frame.imageHandle();
        advance(GeneratedOutputReturnState::EMPTY, GeneratedOutputReturnState::D2_VALIDATED);
        result.frame.consume();
        advance(GeneratedOutputReturnState::D2_VALIDATED, GeneratedOutputReturnState::TOKEN_CONSUMED);

        backing = vk::VulkanNativeExternalImageBacking::create(
            generationEndpoint, renderEndpoint,
            authoritative.extent, authoritative.format);
        advance(GeneratedOutputReturnState::TOKEN_CONSUMED, GeneratedOutputReturnState::B_BACKING_READY);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = generationEndpoint.queueFamilyIndex;
        auto vr = generationEndpoint.CreateCommandPool(generationEndpoint.bufferDevice.device,
            &pool, nullptr, &bCommandPool);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A1 B command pool creation");
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = bCommandPool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        vr = generationEndpoint.AllocateCommandBuffers(generationEndpoint.bufferDevice.device,
            &allocate, &bCommand);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A1 B command allocation");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vr = generationEndpoint.BeginCommandBuffer(bCommand, &begin);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A1 vkBeginCommandBuffer B");
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            generationEndpoint.queueFamilyIndex, backing.image(), range};
        generationEndpoint.CmdPipelineBarrier(bCommand, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDst);
        const VkImageCopy copy{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
            {authoritative.extent.width, authoritative.extent.height, 1}};
        generationEndpoint.CmdCopyImage(bCommand, sourceImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, backing.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        const VkImageMemoryBarrier release{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, generationEndpoint.queueFamilyIndex,
            VK_QUEUE_FAMILY_FOREIGN_EXT, backing.image(), range};
        generationEndpoint.CmdPipelineBarrier(bCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &release);
        vr = generationEndpoint.EndCommandBuffer(bCommand);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A1 vkEndCommandBuffer B");
        signalB = vk::createExportableSyncFdSemaphore(generationEndpoint.semaphoreDevice);
        const VkSemaphore signal = signalB.handle();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1;
        submit.pCommandBuffers = &bCommand; submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &signal;
        // The qualified active route intentionally retains its historical
        // VK_NULL_HANDLE fence.  The inactive shadow route supplies a real
        // fence through the same submission seam; no host wait is introduced.
        vr = generationEndpoint.QueueSubmit(generationEndpoint.queue, 1, &submit,
            returnSubmissionFence(ReturnSubmissionFencePolicy::ACTIVE_COMPATIBLE,
                VK_NULL_HANDLE));
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A1 vkQueueSubmit B");
        bSubmitted = true;
        advance(GeneratedOutputReturnState::B_BACKING_READY, GeneratedOutputReturnState::B_COPY_SUBMITTED);
        auto sync = vk::exportSyncFd(generationEndpoint.semaphoreDevice, signalB.handle());
        advance(GeneratedOutputReturnState::B_COPY_SUBMITTED, GeneratedOutputReturnState::B_SYNC_EXPORTED);
        auto descriptor = backing.exportDescriptor();
        const vk::RuntimeImageBackingInfo info{.extent = descriptor.extent,
            .format = descriptor.format,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .backingSize = descriptor.allocationSize, .fourcc = descriptor.fourcc,
            .modifier = descriptor.modifier,
            .planeCount = static_cast<uint32_t>(descriptor.planes.size()),
            .plane = descriptor.planes.front(), .planes = descriptor.planes};
        importedA = vk::createRuntimeImageEndpoint(renderEndpoint,
            std::move(descriptor.dmaBuf), info);
        importedA = vk::RuntimeImageEndpoint::createExecutionResources(std::move(importedA), 1);
        advance(GeneratedOutputReturnState::B_SYNC_EXPORTED, GeneratedOutputReturnState::A_IMPORTED);
        advance(GeneratedOutputReturnState::A_IMPORTED, GeneratedOutputReturnState::A_COPY_SUBMITTED);
        const auto bytes = vk::RuntimeImageEndpoint::readForeignImage(importedA, std::move(sync));
        aCompleted = true;
        GeneratedOutputIntegrity observed{authoritative.generation, authoritative.extent,
            authoritative.format, bytes.size(), 0,
            lsfgvk::common::fnv1a64(bytes.data(), bytes.size())};
        for (const uint8_t byte : bytes) {
            observed.nonzeroByteCount += byte != 0;
        }
        if (!generatedOutputIntegrityMatches(authoritative, observed))
            throw std::runtime_error("D3A1 same-generation B/A integrity mismatch");
        advance(GeneratedOutputReturnState::A_COPY_SUBMITTED, GeneratedOutputReturnState::A_VALIDATED);
        advance(GeneratedOutputReturnState::A_VALIDATED, GeneratedOutputReturnState::PASS);
        std::cerr << "[DG2X-P4C-D3A1] Generated output B->A diagnostic\n"
            << "  Generation identity: " << authoritative.generation << "\n"
            << "  Intermediate host wait Generate->return: PRESENT — D2 diagnostic completion\n"
            << "  Host wait B->A: NONE\n  CPU frame bridge: NONE\n  Presentation: NONE\n"
            << "  B/A byte count: " << authoritative.byteCount << "\n"
            << "  B/A non-zero bytes: " << authoritative.nonzeroByteCount << "\n"
            << "  B/A checksum: 0x" << std::hex << authoritative.checksum << std::dec << "\n"
            << "DG2X_P4C_D3A1_GENERATED_OUTPUT_B_TO_A_PASS\n";
    } catch (...) {
        // Failure cleanup only: success never waits on B. If import/submission on A
        // fails after B was submitted, complete B before destroying in-use resources.
        if (bSubmitted && !aCompleted && generationEndpoint.DeviceWaitIdle)
            static_cast<void>(generationEndpoint.DeviceWaitIdle(
                generationEndpoint.bufferDevice.device));
        states.fail();
        throw;
    }
}

void GeneratedOutputReturnSession::executeGpuChained(
        backend::RuntimeGenerateDiagnosticPending&& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession,
        std::optional<vk::RuntimeForeignImageHandoffInfo> handoff) {
    try {
        auto bPending = submitGeneratedBReturn(std::move(pending),
            ReturnSubmissionFencePolicy::ACTIVE_COMPATIBLE, backend, backendSession);
        completeGeneratedReturnOnA(std::move(bPending), backend, backendSession, handoff);
    } catch (...) {
        chainedStates.fail();
        throw;
    }
}

RuntimeGeneratedBReturnPending GeneratedOutputReturnSession::submitGeneratedBReturn(
        backend::RuntimeGenerateDiagnosticPending&& pending,
        ReturnSubmissionFencePolicy fencePolicy, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession, VkFence shadowFence) {
    if (!pending.valid() || pending.identity() == 0
            || pending.layoutValue() != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            || pending.queueFamily() != generationEndpoint.queueFamilyIndex
            || pending.readinessSemaphore() == VK_NULL_HANDLE)
        throw std::invalid_argument("invalid or stale D3A2 pending generation capability");
    const auto generation = pending.identity();
    const auto extent = pending.extentValue();
    const auto format = pending.formatValue();
    const auto sourceImage = pending.imageHandle();
    const auto generationReady = pending.readinessSemaphore();
    const bool shadowPolicy = fencePolicy == ReturnSubmissionFencePolicy::SHADOW_REAL;
    std::shared_ptr<VkFence> fenceOwner;
    if (shadowPolicy) {
        VkFence created{};
        const VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        const auto result = generationEndpoint.CreateFence(
            generationEndpoint.bufferDevice.device, &info, nullptr, &created);
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, "D3A3 shadow B-return fence");
        fenceOwner = std::shared_ptr<VkFence>(new VkFence(created),
            [endpoint = generationEndpoint](VkFence* value) {
                if (value && *value && endpoint.DestroyFence)
                    endpoint.DestroyFence(endpoint.bufferDevice.device, *value, nullptr);
                delete value;
            });
        shadowFence = created;
    }
    backend::RuntimeSubmissionRetirement authority;
    std::shared_ptr<void> returnResources;
    bool shadowRejectionRecorded{};
    try {
        chainedStates.advance(GpuChainedReturnState::EMPTY,
            GpuChainedReturnState::D2_SUBMITTED);
        chainedStates.advance(GpuChainedReturnState::D2_SUBMITTED,
            GpuChainedReturnState::PENDING_GENERATION);
        pending.consumeTransport();
        chainedStates.advance(GpuChainedReturnState::PENDING_GENERATION,
            GpuChainedReturnState::TRANSPORT_CONSUMED);

        backing = vk::VulkanNativeExternalImageBacking::create(
            generationEndpoint, renderEndpoint, extent, format);
        chainedStates.advance(GpuChainedReturnState::TRANSPORT_CONSUMED,
            GpuChainedReturnState::RETURN_B_PREPARED);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = generationEndpoint.queueFamilyIndex;
        auto vr = generationEndpoint.CreateCommandPool(generationEndpoint.bufferDevice.device,
            &pool, nullptr, &bCommandPool);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A2 B command pool creation");
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = bCommandPool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        vr = generationEndpoint.AllocateCommandBuffers(generationEndpoint.bufferDevice.device,
            &allocate, &bCommand);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A2 B command allocation");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vr = generationEndpoint.BeginCommandBuffer(bCommand, &begin);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A2 vkBeginCommandBuffer B");
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            generationEndpoint.queueFamilyIndex, backing.image(), range};
        generationEndpoint.CmdPipelineBarrier(bCommand, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDst);
        const VkImageCopy copy{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
            {extent.width, extent.height, 1}};
        generationEndpoint.CmdCopyImage(bCommand, sourceImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, backing.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        const VkImageMemoryBarrier release{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, generationEndpoint.queueFamilyIndex,
            VK_QUEUE_FAMILY_FOREIGN_EXT, backing.image(), range};
        generationEndpoint.CmdPipelineBarrier(bCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &release);
        vr = generationEndpoint.EndCommandBuffer(bCommand);
        if (vr != VK_SUCCESS) throw ls::vulkan_error(vr, "D3A2 vkEndCommandBuffer B");
        signalB = vk::createExportableSyncFdSemaphore(generationEndpoint.semaphoreDevice);
        const VkSemaphore signal = signalB.handle();
        constexpr VkPipelineStageFlags generationWaitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &generationReady;
        submit.pWaitDstStageMask = &generationWaitStage;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &bCommand;
        submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = &signal;
        vr = generationEndpoint.QueueSubmit(generationEndpoint.queue, 1, &submit,
            returnSubmissionFence(fencePolicy, shadowFence));
        if (vr != VK_SUCCESS) {
            if (shadowPolicy) {
                backend.recordRuntimeBReturnRejected(
                    backendSession, generation, vr == VK_ERROR_DEVICE_LOST);
                shadowRejectionRecorded = true;
            }
            throw ls::vulkan_error(vr, "D3A2 vkQueueSubmit B");
        }
        bSubmitted = true;
        chainedStates.advance(GpuChainedReturnState::RETURN_B_PREPARED,
            GpuChainedReturnState::RETURN_B_SUBMITTED);
        if (shadowPolicy) {
            authority = backend::RuntimeSubmissionRetirement::submittedFence(
                shadowFence, generation, operationLifetime);
            backend.recordRuntimeBReturnSubmitted(backendSession, generation);
            auto owned = std::make_shared<ShadowBReturnResourceLifetime>();
            owned->endpoint = generationEndpoint;
            owned->backing = std::move(backing);
            owned->signal = std::move(signalB);
            owned->commandPool = std::exchange(bCommandPool, VK_NULL_HANDLE);
            owned->command = std::exchange(bCommand, VK_NULL_HANDLE);
            returnResources = std::move(owned);
        }
        vk::SyncFdPayload sync;
        try {
            VkSemaphore exportSemaphore = signalB.handle();
            if (shadowPolicy) {
                exportSemaphore = std::static_pointer_cast<ShadowBReturnResourceLifetime>(
                    returnResources)->signal.handle();
            }
            sync = vk::exportSyncFd(generationEndpoint.semaphoreDevice, exportSemaphore);
        } catch (...) {
            if (shadowPolicy && authority.valid()) {
                acceptedShadowFailure.emplace(RuntimeGeneratedBReturnPending(
                    std::move(pending), vk::SyncFdPayload{},
                    std::move(authority), std::move(fenceOwner),
                    std::move(returnResources)));
            }
            throw;
        }
        return RuntimeGeneratedBReturnPending(std::move(pending), std::move(sync),
            std::move(authority), std::move(fenceOwner), std::move(returnResources));
    } catch (...) {
        if (shadowPolicy) {
            if (!bSubmitted && !shadowRejectionRecorded && pending.valid()) {
                try { backend.recordRuntimeBReturnRejected(
                    backendSession, generation, false); }
                catch (...) {}
            }
            throw;
        }
        if (bSubmitted && generationEndpoint.DeviceWaitIdle)
            static_cast<void>(generationEndpoint.DeviceWaitIdle(
                generationEndpoint.bufferDevice.device));
        if (pending.valid()) {
            try { static_cast<void>(backend.completeRuntimeGenerateDiagnostic(
                backendSession, std::move(pending))); }
            catch (...) {}
        }
        throw;
    }
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
RuntimeGeneratedBReturnPending
GeneratedOutputReturnSession::submitShadowBReturnForTesting(
        backend::RuntimeGenerateDiagnosticPending&& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    return submitProductionBReturn(std::move(pending), backend, backendSession);
}
#endif

void GeneratedOutputReturnSession::completeGeneratedReturnOnA(
        RuntimeGeneratedBReturnPending&& bPending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession,
        std::optional<vk::RuntimeForeignImageHandoffInfo> handoff) {
    auto& pending = bPending.pending;
    if (!pending.valid() || !bPending.bToAPayload.valid())
        throw std::invalid_argument("invalid B-return pending for A completion");
    const bool boundedShadowSubmission = bPending.bReturn.valid();
    auto sync = std::move(bPending.bToAPayload);
    const auto generation = pending.identity();
    const auto extent = pending.extentValue();
    const auto format = pending.formatValue();
    try {
        auto* returnBacking = &backing;
        if (boundedShadowSubmission) {
            if (!bPending.returnResources)
                throw std::logic_error("shadow B-return resources expired before A import");
            returnBacking = &std::static_pointer_cast<ShadowBReturnResourceLifetime>(
                bPending.returnResources)->backing;
        }
        auto descriptor = returnBacking->exportDescriptor();
        const vk::RuntimeImageBackingInfo info{.extent = descriptor.extent,
            .format = descriptor.format,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .backingSize = descriptor.allocationSize, .fourcc = descriptor.fourcc,
            .modifier = descriptor.modifier,
            .planeCount = static_cast<uint32_t>(descriptor.planes.size()),
            .plane = descriptor.planes.front(), .planes = descriptor.planes};
        importedA = vk::createRuntimeImageEndpoint(renderEndpoint,
            std::move(descriptor.dmaBuf), info);
        importedA = vk::RuntimeImageEndpoint::createExecutionResources(std::move(importedA), 1);
        if (handoff.has_value()) {
            aReadbackPending = vk::RuntimeImageEndpoint::submitForeignImageReadback(
                std::move(importedA), std::move(sync), handoff);
            chainedStates.advance(GpuChainedReturnState::RETURN_B_SUBMITTED,
                GpuChainedReturnState::A_SUBMITTED);
            expected = {generation, extent, format,
                static_cast<size_t>(extent.width) * extent.height * 4U, 0, 0};
            delayedGeneration = std::move(pending);
            delayedBackend = &backend;
            delayedBackendSession = &backendSession;
            return;
        }
        const auto bytes = vk::RuntimeImageEndpoint::readForeignImage(importedA, std::move(sync));
        aCompleted = true;
        chainedStates.advance(GpuChainedReturnState::RETURN_B_SUBMITTED,
            GpuChainedReturnState::A_SUBMITTED);
        chainedStates.advance(GpuChainedReturnState::A_SUBMITTED,
            GpuChainedReturnState::A_COMPLETED);

        auto validated = backend.completeRuntimeGenerateDiagnostic(
            backendSession, std::move(pending));
        authoritative = {validated.metadata.generation, validated.metadata.extent,
            validated.metadata.format, validated.metadata.byteCount,
            validated.metadata.nonzeroByteCount, validated.metadata.checksum};
        chainedStates.advance(GpuChainedReturnState::A_COMPLETED,
            GpuChainedReturnState::B_VALIDATED);
        GeneratedOutputIntegrity observed{generation, extent, format, bytes.size(), 0,
            lsfgvk::common::fnv1a64(bytes.data(), bytes.size())};
        for (const uint8_t byte : bytes) observed.nonzeroByteCount += byte != 0;
        chainedStates.advance(GpuChainedReturnState::B_VALIDATED,
            GpuChainedReturnState::A_VALIDATED);
        const GpuChainedMarkerGate gate{true, false, false, true, true,
            generatedOutputIntegrityMatches(authoritative, observed), false};
        if (!gate.ready())
            throw std::runtime_error("D3A2 same-generation B/A integrity mismatch");
        chainedStates.advance(GpuChainedReturnState::A_VALIDATED,
            GpuChainedReturnState::PASS);
        std::cerr << "[DG2X-P4C-D3A2] GPU-chained generated output B->A diagnostic\n"
            << "  Generation: " << authoritative.generation << "\n"
            << "  Generate->return dependency: B_GPU_SEMAPHORE\n"
            << "  Intermediate host wait Generate->return: NONE\n"
            << "  Host wait B->A: NONE\n"
            << "  Final diagnostic host completion: PRESENT\n"
            << "  B diagnostic validation: PASS\n  A returned validation: PASS\n"
            << "  Same-generation integrity: PASS\n"
            << "  Backend execution: GENERATE_DIAGNOSTIC\n"
            << "  Input transport: A_TO_B_GPU_SYNC\n"
            << "  Output transport: B_TO_A_GPU_SYNC\n"
            << "  Presentation: NONE\n  CPU frame bridge: NONE\n"
            << "DG2X_P4C_D3A2_GPU_CHAINED_GENERATED_OUTPUT_B_TO_A_PASS\n";
    } catch (...) {
        if (!boundedShadowSubmission && pending.valid()) {
            try { static_cast<void>(backend.completeRuntimeGenerateDiagnostic(
                backendSession, std::move(pending))); }
            catch (...) {}
        }
        throw;
    }
}

RuntimeGeneratedBReturnPending
GeneratedOutputReturnSession::submitProductionBReturn(
        backend::RuntimeGenerateDiagnosticPending&& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    try {
        return submitGeneratedBReturn(std::move(pending),
            ReturnSubmissionFencePolicy::SHADOW_REAL, backend, backendSession);
    } catch (...) {
        if (!acceptedShadowFailure) chainedStates.fail();
        throw;
    }
}

lsfgvk::backend::ReturnedGeneratedOperation
GeneratedOutputReturnSession::completeProductionGeneratedReturnOnA(
        RuntimeGeneratedBReturnPending&& bPending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession,
        vk::RuntimeForeignImageHandoffInfo handoff) {
    if (!bPending.valid() || !bPending.bReturn.valid()
            || handoff.destinationQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED
            || handoff.signalSemaphore == VK_NULL_HANDLE)
        throw std::invalid_argument("invalid production B-return/A-handoff composition");
    const auto identity = bPending.identity();
    try {
        completeGeneratedReturnOnA(std::move(bPending), backend, backendSession, handoff);
    } catch (...) {
        if (bPending.bReturn.valid()) acceptedShadowFailure.emplace(std::move(bPending));
        throw;
    }
    if (!aReadbackPending.valid() || !delayedGeneration.valid()
            || aReadbackPending.returnedSignalSemaphore() != handoff.signalSemaphore
            || aReadbackPending.retirementFence() == VK_NULL_HANDLE
            || aReadbackPending.importedWaitSemaphore() == VK_NULL_HANDLE)
        throw std::logic_error("production A-return did not preserve bounded identities");

    const auto generation = identity.generationId;
    auto aAuthority = backend::RuntimeSubmissionRetirement::submittedFence(
        aReadbackPending.retirementFence(), generation, operationLifetime);
    auto payloadAuthority = backend::RuntimeTemporarySemaphorePayload::submittedWait(
        aReadbackPending.importedWaitSemaphore(), generation, operationLifetime);
    auto generatedAuthority = delayedGeneration.operationRetirementAuthority();
    backend.recordRuntimeAReturnSubmitted(backendSession, generation);
    auto result = backend::ReturnedGeneratedOperation(identity,
        std::move(bPending.bReturn), std::move(aAuthority),
        std::move(aReadbackPending), std::move(payloadAuthority),
        std::move(delayedGeneration), std::move(generatedAuthority),
        handoff.signalSemaphore, std::move(bPending.fenceOwner),
        std::move(bPending.returnResources), operationLifetime);
    delayedBackend = nullptr;
    delayedBackendSession = nullptr;
    return result;
}

lsfgvk::backend::RuntimeRetirementStatus
GeneratedOutputReturnSession::tryRetireProductionBReturn(
        RuntimeGeneratedBReturnPending& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    const auto generation = pending.identity().generationId;
    auto& authority = pending.bReturn;
    if (authority.state() == backend::RuntimeAuthorityState::RETIRED)
        return backend::RuntimeRetirementStatus::RETIRED;
    if (!pending.valid() || authority.state() != backend::RuntimeAuthorityState::SUBMITTED
            || authority.epoch() != generation || !generationEndpoint.GetFenceStatus)
        throw std::logic_error("invalid production pending B-return retirement request");
    const auto result = generationEndpoint.GetFenceStatus(
        generationEndpoint.bufferDevice.device, authority.fenceHandle());
    if (result == VK_NOT_READY)
        return backend::RuntimeRetirementStatus::NOT_READY;
    if (result != VK_SUCCESS) {
        authority.fail();
        backend.failRuntimeBReturnRetirement(
            backendSession, generation, result == VK_ERROR_DEVICE_LOST);
        return result == VK_ERROR_DEVICE_LOST
            ? backend::RuntimeRetirementStatus::DEVICE_LOST
            : backend::RuntimeRetirementStatus::FAILED;
    }
    backend.retireRuntimeBReturnWait(backendSession, generation);
    authority.retire(generation);
    return backend::RuntimeRetirementStatus::RETIRED;
}

lsfgvk::backend::RuntimeRetirementStatus
GeneratedOutputReturnSession::tryRetireProductionBReturn(
        backend::ReturnedGeneratedOperation& operation, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    const auto generation = operation.identity().generationId;
    auto& authority = operation.bReturnAuthority();
    if (authority.state() == backend::RuntimeAuthorityState::RETIRED)
        return backend::RuntimeRetirementStatus::RETIRED;
    if (!operation.valid() || authority.state() != backend::RuntimeAuthorityState::SUBMITTED
            || authority.epoch() != generation || !generationEndpoint.GetFenceStatus)
        throw std::logic_error("invalid production B-return retirement request");
    const auto result = generationEndpoint.GetFenceStatus(
        generationEndpoint.bufferDevice.device, authority.fenceHandle());
    if (result == VK_NOT_READY)
        return backend::RuntimeRetirementStatus::NOT_READY;
    if (result != VK_SUCCESS) {
        authority.fail();
        backend.failRuntimeBReturnRetirement(
            backendSession, generation, result == VK_ERROR_DEVICE_LOST);
        return result == VK_ERROR_DEVICE_LOST
            ? backend::RuntimeRetirementStatus::DEVICE_LOST
            : backend::RuntimeRetirementStatus::FAILED;
    }
    backend.retireRuntimeBReturnWait(backendSession, generation);
    authority.retire(generation);
    return backend::RuntimeRetirementStatus::RETIRED;
}

lsfgvk::backend::RuntimeRetirementStatus
GeneratedOutputReturnSession::releaseProductionGeneratedOutput(
        backend::ReturnedGeneratedOperation& operation, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    if (operation.generatedOutputRetired())
        return backend::RuntimeRetirementStatus::RETIRED;
    if (!operation.valid())
        throw std::logic_error("invalid production generated-output release request");
    if (operation.bReturnAuthority().state() != backend::RuntimeAuthorityState::RETIRED)
        return backend::RuntimeRetirementStatus::NOT_READY;
    try {
        operation.rejectGeneratedOutputRetirement();
        if (!operation.generatedOutput.valid() || !operation.generatedRetirement.valid()
                || operation.generatedRetirement.generationId()
                    != operation.identity().generationId)
            throw std::logic_error("generated-output retirement authority is missing");
        static_cast<void>(backend.completeRuntimeGenerateDiagnostic(
            backendSession, std::move(operation.generatedOutput)));
        backend.retireRuntimeGenerateOperation(
            backendSession, std::move(operation.generatedRetirement));
        backend.releaseRuntimeGenerationReady(
            backendSession, operation.identity().generationId);
        operation.markGeneratedOutputRetired();
        return backend::RuntimeRetirementStatus::RETIRED;
    } catch (...) {
        return backend::RuntimeRetirementStatus::FAILED;
    }
}

void GeneratedOutputReturnSession::resetAfterProductionHandoff() {
    if (aReadbackPending.valid() || delayedGeneration.valid()
            || delayedBackend || delayedBackendSession)
        throw std::logic_error("D3A2 production authority remains in the return session");
    chainedStates.resetAfterProductionHandoff();
    operationLifetime = std::make_shared<const uint8_t>(0);
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
lsfgvk::backend::ReturnedGeneratedOperation
GeneratedOutputReturnSession::completeShadowGeneratedReturnOnAForTesting(
        RuntimeGeneratedBReturnPending&& bPending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession,
        vk::RuntimeForeignImageHandoffInfo handoff) {
    return completeProductionGeneratedReturnOnA(
        std::move(bPending), backend, backendSession, handoff);
}

void GeneratedOutputReturnSession::retireShadowBReturnForTesting(
        backend::ReturnedGeneratedOperation& operation, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    const auto generation = operation.identity().generationId;
    auto& authority = operation.bReturnAuthority();
    if (!operation.valid() || authority.state() != backend::RuntimeAuthorityState::SUBMITTED
            || authority.epoch() != generation || !generationEndpoint.WaitForFences)
        throw std::logic_error("invalid shadow B-return fence retirement request");
    const VkFence fence = authority.fenceHandle();
    const auto waitResult = generationEndpoint.WaitForFences(
        generationEndpoint.bufferDevice.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS) {
        if (waitResult != VK_TIMEOUT) {
            authority.fail();
            backend.failShadowBReturnRetirement(
                backendSession, generation, waitResult == VK_ERROR_DEVICE_LOST);
        }
        throw ls::vulkan_error(waitResult, "shadow B-return fence retirement failed");
    }
    const auto result = tryRetireProductionBReturn(operation, backend, backendSession);
    if (result != backend::RuntimeRetirementStatus::RETIRED)
        throw backend::error("shadow B-return retirement did not complete");
}

void GeneratedOutputReturnSession::releaseShadowGeneratedOutputForTesting(
        backend::ReturnedGeneratedOperation& operation, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    const auto result = releaseProductionGeneratedOutput(operation, backend, backendSession);
    if (result != backend::RuntimeRetirementStatus::RETIRED)
        throw backend::error("shadow generated-output release did not complete");
}

void GeneratedOutputReturnSession::retireShadowAReturnForTesting(
        backend::ReturnedGeneratedOperation& operation) {
    if (!operation.valid()
            || operation.aReturnSubmission.state()
                != backend::RuntimeAuthorityState::SUBMITTED)
        throw std::logic_error("invalid shadow A-return retirement request");
    try {
        static_cast<void>(vk::RuntimeImageEndpoint::completeForeignImageReadback(
            operation.aReturn));
        operation.retireAReturn(operation.identity().generationId);
    } catch (...) {
        operation.failAReturn();
        throw;
    }
}

std::optional<RuntimeGeneratedBReturnPending>
GeneratedOutputReturnSession::takeAcceptedShadowFailureForTesting() {
    if (!acceptedShadowFailure)
        throw std::logic_error("no accepted shadow B-return failure authority");
    auto result = std::move(acceptedShadowFailure);
    acceptedShadowFailure.reset();
    return result;
}

void GeneratedOutputReturnSession::retireAcceptedShadowBReturnFailureForTesting(
        RuntimeGeneratedBReturnPending& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    if (!pending.acceptedSubmissionAuthorityForTesting()
            || pending.bReturn.state() != backend::RuntimeAuthorityState::SUBMITTED
            || !generationEndpoint.WaitForFences)
        throw std::logic_error("invalid accepted shadow failure retirement request");
    const auto generation = pending.pending.identity();
    const VkFence fence = pending.bReturn.fenceHandle();
    const auto result = generationEndpoint.WaitForFences(
        generationEndpoint.bufferDevice.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        if (result != VK_TIMEOUT) {
            pending.bReturn.fail();
            backend.failShadowBReturnRetirement(
                backendSession, generation, result == VK_ERROR_DEVICE_LOST);
        }
        throw ls::vulkan_error(result, "accepted shadow B-return fence retirement failed");
    }
    backend.retireShadowBReturnWait(backendSession, generation);
    pending.bReturn.retire(generation);
}

void GeneratedOutputReturnSession::releaseAcceptedShadowGeneratedOutputForTesting(
        RuntimeGeneratedBReturnPending& pending, backend::Instance& backend,
        backend::RuntimeGenerateSession& backendSession) {
    if (pending.bReturn.state() != backend::RuntimeAuthorityState::RETIRED
            || !pending.pending.valid())
        throw std::logic_error("accepted shadow output still has an in-flight reader");
    const auto generation = pending.pending.identity();
    auto generatedAuthority = pending.pending.operationRetirementAuthority();
    static_cast<void>(backend.completeRuntimeGenerateDiagnostic(
        backendSession, std::move(pending.pending)));
    backend.retireRuntimeGenerateOperation(backendSession, std::move(generatedAuthority));
    backend.releaseShadowGenerationReady(backendSession, generation);
}
#endif

void GeneratedOutputReturnSession::completePresentationDiagnostics() {
    if (!aReadbackPending.valid() || !delayedGeneration.valid()
            || !delayedBackend || !delayedBackendSession)
        throw std::logic_error("D3B1 presentation diagnostics are not pending");
    try {
        const auto bytes = vk::RuntimeImageEndpoint::completeForeignImageReadback(
            aReadbackPending);
        aCompleted = true;
        chainedStates.advance(GpuChainedReturnState::A_SUBMITTED,
            GpuChainedReturnState::A_COMPLETED);
        auto validated = delayedBackend->completeRuntimeGenerateDiagnostic(
            *delayedBackendSession, std::move(delayedGeneration));
        authoritative = {validated.metadata.generation, validated.metadata.extent,
            validated.metadata.format, validated.metadata.byteCount,
            validated.metadata.nonzeroByteCount, validated.metadata.checksum};
        chainedStates.advance(GpuChainedReturnState::A_COMPLETED,
            GpuChainedReturnState::B_VALIDATED);
        GeneratedOutputIntegrity observed{expected.generation, expected.extent,
            expected.format, bytes.size(), 0,
            lsfgvk::common::fnv1a64(bytes.data(), bytes.size())};
        for (const uint8_t byte : bytes) observed.nonzeroByteCount += byte != 0;
        chainedStates.advance(GpuChainedReturnState::B_VALIDATED,
            GpuChainedReturnState::A_VALIDATED);
        if (!generatedOutputIntegrityMatches(authoritative, observed))
            throw std::runtime_error("D3B1 same-generation B/A integrity mismatch");
        std::cerr << "[DG2X-P4C-D3B1] terminal returned-image integrity\n"
            << "  B generation: " << authoritative.generation << "\n"
            << "  B format: " << static_cast<int>(authoritative.format) << "\n"
            << "  B extent: " << authoritative.extent.width << 'x'
            << authoritative.extent.height << "\n"
            << "  B bytes: " << authoritative.byteCount << "\n"
            << "  B nonzero bytes: " << authoritative.nonzeroByteCount << "\n"
            << "  B FNV-1a: 0x" << std::hex << authoritative.checksum << std::dec << "\n"
            << "  A generation: " << observed.generation << "\n"
            << "  A format: " << static_cast<int>(observed.format) << "\n"
            << "  A extent: " << observed.extent.width << 'x' << observed.extent.height << "\n"
            << "  A bytes: " << observed.byteCount << "\n"
            << "  A nonzero bytes: " << observed.nonzeroByteCount << "\n"
            << "  A FNV-1a: 0x" << std::hex << observed.checksum << std::dec << "\n";
        chainedStates.advance(GpuChainedReturnState::A_VALIDATED,
            GpuChainedReturnState::PASS);
    } catch (...) {
        chainedStates.fail();
        throw;
    }
}
