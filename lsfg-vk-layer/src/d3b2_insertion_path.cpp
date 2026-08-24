#include "d3b2_insertion_path.hpp"

#include "lsfg-vk-common/helpers/errors.hpp"

#include <stdexcept>
#include <array>

namespace lsfgvk::layer {

void validateD3B2InsertionPreflight(const D3B2InsertionPath& path) {
    if (!path.singleSwapchain || path.hasPNext || !path.fifo
            || !path.generated.image || !path.original.image
            || !path.generated.blitSourceSupported
            || !path.original.blitSourceSupported
            || !path.hiddenBlitDestinationSupported
            || !path.originalReady || !path.returnedForGraphics
            || !path.acquireGenerated || !path.acquireOriginal
            || !path.releaseAcquiredImages
            || !path.maintenanceReleaseCapable
            || !path.generatedPresentReady || !path.originalPresentReady
            || !path.graphicsFence
            || path.commandPoolFamily == VK_QUEUE_FAMILY_IGNORED
            || path.commandPoolFamily != path.submitQueueFamily
            || !(path.submitQueueFlags & VK_QUEUE_GRAPHICS_BIT)
            || (!path.submit && !path.submitWithWaitStage)
            || path.returnedForGraphicsWaitStage
                != d3b2FirstTerminalConsumerStage())
        throw std::runtime_error("D3B2 insertion preflight failed");
}

VkResult executeD3B2Insertion(D3B2InsertionPath& path) {
    if (!path.state || *path.state == D3B2InsertionState::PASS
            || *path.state == D3B2InsertionState::FAILED)
        throw std::logic_error("D3B2 insertion is not re-entrant");
    bool submitted = false;
    bool deviceLost = false;
    std::array<D3B2ImageState, 2> imageStates{
        D3B2ImageState::NOT_ACQUIRED, D3B2ImageState::NOT_ACQUIRED};
    std::array<uint32_t, 2> imageIndices{};
    bool retired = false;
    const auto retire = [&] { if (!retired) { path.retire(); retired = true; } };
    try {
        *path.state = D3B2InsertionState::PREFLIGHT;
        validateD3B2InsertionPreflight(path);
        *path.state = D3B2InsertionState::GENERATED_READY;
        const auto generated = path.acquire(path.acquireGenerated);
        imageIndices[0] = generated.index;
        imageStates[0] = D3B2ImageState::ACQUIRED_IDLE;
        const auto original = path.acquire(path.acquireOriginal);
        imageIndices[1] = original.index;
        imageStates[1] = D3B2ImageState::ACQUIRED_IDLE;
        if (!generated.image || !original.image || generated.index == original.index)
            throw std::runtime_error("D3B2 hidden acquire invariant failed");
        path.distinctHiddenImages = true;
        if (path.onDestinationsAcquired) path.onDestinationsAcquired();
        *path.state = D3B2InsertionState::TWO_ACQUIRES_READY;
        if (path.onRecordBegin) path.onRecordBegin();
        path.record(generated, original);
        if (path.onRecordEnd) path.onRecordEnd();
        const auto submitResult = path.submitWithWaitStage
            ? path.submitWithWaitStage(path.returnedForGraphicsWaitStage)
            : path.submit();
        if (submitResult != VK_SUCCESS)
        {
            deviceLost = submitResult == VK_ERROR_DEVICE_LOST;
            throw ls::vulkan_error(submitResult, "D3B2 graphics submit failed");
        }
        submitted = true;
        imageStates.fill(D3B2ImageState::SUBMITTED_FOR_GRAPHICS);
        *path.state = D3B2InsertionState::INSERT_SUBMITTED;
        if (path.onTerminalSubmitAccepted) path.onTerminalSubmitAccepted();
        imageStates[0] = D3B2ImageState::PRESENT_CALL_ISSUED;
        const auto generatedResult = path.presentFences.enabled() && path.presentWithFence
            ? path.presentWithFence(generated, path.generatedPresentReady,
                path.presentFences.generated)
            : path.present(generated, path.generatedPresentReady);
        if (generatedResult != VK_SUCCESS)
        {
            deviceLost = generatedResult == VK_ERROR_DEVICE_LOST;
            throw ls::vulkan_error(generatedResult, "D3B2 generated present failed");
        }
        imageStates[0] = D3B2ImageState::PRESENT_ACQUISITION_RELEASED;
        *path.state = D3B2InsertionState::GENERATED_PRESENTED;
        if (path.onGeneratedPresentAccepted) path.onGeneratedPresentAccepted();
        imageStates[1] = D3B2ImageState::PRESENT_CALL_ISSUED;
        const auto originalResult = path.presentFences.enabled() && path.presentWithFence
            ? path.presentWithFence(original, path.originalPresentReady,
                path.presentFences.original)
            : path.present(original, path.originalPresentReady);
        if (originalResult != VK_SUCCESS)
        {
            deviceLost = originalResult == VK_ERROR_DEVICE_LOST;
            throw ls::vulkan_error(originalResult, "D3B2 original present failed");
        }
        imageStates[1] = D3B2ImageState::PRESENT_ACQUISITION_RELEASED;
        *path.state = D3B2InsertionState::ORIGINAL_PRESENTED;
        if (path.onOriginalPresentAccepted) path.onOriginalPresentAccepted();
        if (!path.waitGraphicsFence())
            throw ls::vulkan_error(VK_TIMEOUT, "D3B2 graphics fence failed");
        submitted = false;
        if (path.onGraphicsFenceRetired) path.onGraphicsFenceRetired();
        *path.state = D3B2InsertionState::VALIDATING;
        if (!path.generatedIntegrity() || !path.originalIdentity())
            throw std::runtime_error("D3B2 terminal validation failed");
        retire();
        *path.state = D3B2InsertionState::PASS;
        path.emitMarker();
        return originalResult;
    } catch (...) {
        *path.state = D3B2InsertionState::FAILED;
        bool fenceSafe = !submitted;
        if (submitted && !deviceLost) {
            try { fenceSafe = path.waitGraphicsFence(); } catch (...) { fenceSafe = false; }
        }
        if (fenceSafe && !deviceLost) {
            std::vector<uint32_t> outstanding;
            for (size_t i = 0; i < imageStates.size(); ++i) {
                if (imageStates[i] == D3B2ImageState::ACQUIRED_IDLE
                        || imageStates[i] == D3B2ImageState::SUBMITTED_FOR_GRAPHICS)
                    outstanding.push_back(imageIndices[i]);
            }
            if (!outstanding.empty()) {
                const auto result = path.releaseAcquiredImages(outstanding);
                if (result == VK_SUCCESS) {
                    for (size_t i = 0; i < imageStates.size(); ++i)
                        if (imageStates[i] == D3B2ImageState::ACQUIRED_IDLE
                                || imageStates[i] == D3B2ImageState::SUBMITTED_FOR_GRAPHICS)
                            imageStates[i] = D3B2ImageState::RELEASED_WITH_MAINTENANCE1;
                }
            }
        }
        retire();
        throw;
    }
}

}
