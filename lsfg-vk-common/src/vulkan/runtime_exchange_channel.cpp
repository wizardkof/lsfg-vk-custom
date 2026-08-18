/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include <vulkan/vulkan_core.h>

using namespace vk;

namespace {
    class FinalFence {
    public:
        FinalFence(const RuntimeExchangeEndpoint& endpoint) : endpoint(&endpoint) {
            const VkFenceCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
            };
            const auto result = endpoint.CreateFence(
                endpoint.bufferDevice.device, &info, nullptr, &this->handle);
            if (result != VK_SUCCESS)
                throw ls::vulkan_error(result, "vkCreateFence() failed for runtime exchange channel");
        }

        FinalFence(const FinalFence&) = delete;
        FinalFence& operator=(const FinalFence&) = delete;
        ~FinalFence() {
            if (this->endpoint && this->handle != VK_NULL_HANDLE)
                this->endpoint->DestroyFence(
                    this->endpoint->bufferDevice.device, this->handle, nullptr);
        }

        [[nodiscard]] VkFence get() const noexcept { return this->handle; }

    private:
        const RuntimeExchangeEndpoint* endpoint{};
        VkFence handle{};
    };

    void validateEndpoint(const RuntimeExchangeEndpoint& endpoint, const char* label) {
        const bool valid = endpoint.bufferDevice.device != VK_NULL_HANDLE
            && endpoint.semaphoreDevice.device == endpoint.bufferDevice.device
            && endpoint.queue != VK_NULL_HANDLE
            && endpoint.QueueSubmit
            && endpoint.CreateFence
            && endpoint.DestroyFence
            && endpoint.WaitForFences
            && endpoint.DeviceWaitIdle;
        if (!valid)
            throw std::invalid_argument(std::string("invalid runtime exchange endpoint: ") + label);
    }

    void submitSignal(const RuntimeExchangeEndpoint& endpoint, VkSemaphore signal) {
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal
        };
        const auto result = endpoint.QueueSubmit(endpoint.queue, 1, &submit, VK_NULL_HANDLE);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "vkQueueSubmit() signal failed for runtime exchange channel");
    }

    void submitWaitSignal(const RuntimeExchangeEndpoint& endpoint,
            VkSemaphore wait, VkSemaphore signal) {
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait,
            .pWaitDstStageMask = &waitStage,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal
        };
        const auto result = endpoint.QueueSubmit(endpoint.queue, 1, &submit, VK_NULL_HANDLE);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "vkQueueSubmit() wait/signal failed for runtime exchange channel");
    }

    void submitFinalWait(const RuntimeExchangeEndpoint& endpoint,
            VkSemaphore wait, VkFence fence) {
        constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait,
            .pWaitDstStageMask = &waitStage
        };
        const auto result = endpoint.QueueSubmit(endpoint.queue, 1, &submit, fence);
        if (result != VK_SUCCESS)
            throw ls::vulkan_error(result, "vkQueueSubmit() final wait failed for runtime exchange channel");
    }
}

RuntimeExchangeChannel::RuntimeExchangeChannel(
        RuntimeExchangeEndpoint render,
        RuntimeExchangeEndpoint generation,
        ImportedExternalBuffer renderBuffer,
        ImportedExternalBuffer generationBuffer,
        SyncFdSemaphore renderExport,
        SyncFdSemaphore renderImport,
        SyncFdSemaphore generationExport,
        SyncFdSemaphore generationImport) noexcept :
    renderEndpoint(std::move(render)),
    generationEndpoint(std::move(generation)),
    renderImportedBuffer(std::move(renderBuffer)),
    generationImportedBuffer(std::move(generationBuffer)),
    renderExportSemaphore(std::move(renderExport)),
    renderImportSemaphore(std::move(renderImport)),
    generationExportSemaphore(std::move(generationExport)),
    generationImportSemaphore(std::move(generationImport)) {
}

RuntimeExchangeChannel vk::createRuntimeExchangeChannel(
        const RuntimeDevicePair& pair,
        RuntimeExchangeEndpoint render,
        RuntimeExchangeEndpoint generation,
        ls::OwnedFd renderFd,
        ls::OwnedFd generationFd,
        const RuntimeExchangeChannelInfo& info) {
    if (!pair.crossDevice())
        throw std::invalid_argument(
            "runtime cross-device exchange channel requires CROSS_PHYSICAL_DEVICE mode");
    validateEndpoint(render, "render");
    validateEndpoint(generation, "generation");
    if (info.logicalSize == 0 || info.backingSize < info.logicalSize || info.usage == 0)
        throw std::invalid_argument("invalid runtime exchange channel buffer contract");

    const ExternalBufferImportInfo importInfo{
        .logicalSize = info.logicalSize,
        .backingSize = info.backingSize,
        .usage = info.usage,
        .requiredMemoryProperties = 0,
        .preferredMemoryProperties = 0
    };

    auto renderBuffer = importDmaBufBuffer(
        render.bufferDevice, std::move(renderFd), importInfo);
    auto generationBuffer = importDmaBufBuffer(
        generation.bufferDevice, std::move(generationFd), importInfo);

    auto renderExport = createExportableSyncFdSemaphore(render.semaphoreDevice);
    auto renderImport = createSyncFdImportSemaphore(render.semaphoreDevice);
    auto generationExport = createExportableSyncFdSemaphore(generation.semaphoreDevice);
    auto generationImport = createSyncFdImportSemaphore(generation.semaphoreDevice);

    return RuntimeExchangeChannel(
        std::move(render), std::move(generation),
        std::move(renderBuffer), std::move(generationBuffer),
        std::move(renderExport), std::move(renderImport),
        std::move(generationExport), std::move(generationImport));
}

RuntimeExchangeSyncDiagnostics RuntimeExchangeChannel::validateSyncRoundTrip() {
    // Keep the final fence alive across failure cleanup so it is never destroyed
    // while a successful final submit could still reference it.
    FinalFence finalFence(this->renderEndpoint);
    bool renderSubmitted = false;
    bool generationSubmitted = false;
    try {
        // A -> B. No host wait is allowed between these submissions.
        submitSignal(this->renderEndpoint, this->renderExportSemaphore.handle());
        renderSubmitted = true;
        auto renderPayload = exportSyncFd(
            this->renderEndpoint.semaphoreDevice, this->renderExportSemaphore.handle());
        const bool renderSentinel = renderPayload.sentinel();
        importSyncFdTemporary(this->generationEndpoint.semaphoreDevice,
            this->generationImportSemaphore.handle(), renderPayload);

        submitWaitSignal(this->generationEndpoint,
            this->generationImportSemaphore.handle(),
            this->generationExportSemaphore.handle());
        generationSubmitted = true;
        auto generationPayload = exportSyncFd(
            this->generationEndpoint.semaphoreDevice,
            this->generationExportSemaphore.handle());
        const bool generationSentinel = generationPayload.sentinel();
        importSyncFdTemporary(this->renderEndpoint.semaphoreDevice,
            this->renderImportSemaphore.handle(), generationPayload);

        submitFinalWait(this->renderEndpoint,
            this->renderImportSemaphore.handle(), finalFence.get());

        // First and only host wait in the successful round trip: all A -> B -> A
        // submits already exist. DeviceWaitIdle is reserved for exceptional cleanup.
        const VkFence fence = finalFence.get();
        const auto waitResult = this->renderEndpoint.WaitForFences(
            this->renderEndpoint.bufferDevice.device,
            1, &fence, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS)
            throw ls::vulkan_error(waitResult,
                "vkWaitForFences() failed for runtime exchange channel");

        return {
            .renderToGenerationSentinel = renderSentinel,
            .generationToRenderSentinel = generationSentinel,
            .hostWaitBeforeFinalSubmit = false
        };
    } catch (...) {
        // Do not destroy semaphores/buffers while submitted work can still reference
        // them. These waits are failure cleanup only and are never the success bridge.
        if (generationSubmitted)
            static_cast<void>(this->generationEndpoint.DeviceWaitIdle(
                this->generationEndpoint.bufferDevice.device));
        if (renderSubmitted)
            static_cast<void>(this->renderEndpoint.DeviceWaitIdle(
                this->renderEndpoint.bufferDevice.device));
        throw;
    }
}
