#include "d3b3_production_seams.hpp"

#include <cassert>
#include <memory>
#include <vector>

using namespace lsfgvk::layer;

template<class T> T h(uintptr_t n) { return reinterpret_cast<T>(n); }

int main() {
    int ingestCalls = 0;
    D3B3IngestDispatch ingest{
        .execute = [&](const CurrentOriginalFrame& frame, size_t slot) {
            ++ingestCalls;
            return RuntimeFrameIngestResult{
                .source = {h<VkImage>(10 + slot), slot, frame.frameId, 1, true},
                .originalReady = h<VkSemaphore>(11), .transportReady = h<VkSemaphore>(12),
                .completionFence = h<VkFence>(13), .temporaryPayloadRetained = true};
        }};
    const CurrentOriginalFrame original{h<VkImage>(1), VK_FORMAT_B8G8R8A8_UNORM,
        {640, 480}, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, 4, 7};
    const auto ingested = ingestCurrentFrame(ingest, original, 1);
    assert(ingestCalls == 1 && ingested.source.slot == 1 && ingested.source.frameId == 7);

    int generateCalls = 0;
    D3B3GenerateDispatch generate{
        .execute = [&](const TemporalSourceRef& older, const TemporalSourceRef& newer,
                uint64_t generation, float interpolation) {
            ++generateCalls;
            return older.slot == 1 && newer.slot == 0 && generation == 2
                && interpolation == 0.5F ? VK_SUCCESS : VK_ERROR_UNKNOWN;
        }};
    const TemporalSourceRef older{h<VkImage>(2), 1, 2, 2, true};
    const TemporalSourceRef newer{h<VkImage>(3), 0, 3, 2, true};
    const auto identity = generateTemporalPair(generate, older, newer, 2);
    assert(generateCalls == 1 && identity.olderFrameId == 2
        && identity.newerFrameId == 3 && identity.generationId == 2);

    int returnCalls = 0;
    D3B3ReturnDispatch returned{
        .execute = [&](GeneratedPairIdentity pair) -> std::optional<ReturnedGeneratedFrame> {
            ++returnCalls;
            return ReturnedGeneratedFrame{h<VkImage>(4), VK_FORMAT_R8G8B8A8_UNORM,
                {640, 480}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 0,
                h<VkSemaphore>(5), h<VkFence>(6), pair,
                std::make_shared<const uint8_t>(0)};
        }};
    const auto output = returnGeneratedFrame(returned, identity);
    assert(returnCalls == 1 && output.identity.newerFrameId == 3);

    D3B2InsertionState state = D3B2InsertionState::INACTIVE;
    std::vector<int> events;
    D3B2InsertionPath path{
        .generated = {.image = h<VkImage>(20), .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {1, 1}, .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .sourceQueueFamily = 0, .destinationQueueFamily = 0, .blitSourceSupported = true},
        .original = {.image = h<VkImage>(21), .format = VK_FORMAT_B8G8R8A8_UNORM,
            .extent = {1, 1}, .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .sourceQueueFamily = 0, .destinationQueueFamily = 0, .blitSourceSupported = true},
        .originalReady = h<VkSemaphore>(22), .returnedForGraphics = h<VkSemaphore>(23),
        .acquireGenerated = h<VkSemaphore>(24), .acquireOriginal = h<VkSemaphore>(25),
        .generatedPresentReady = h<VkSemaphore>(26), .originalPresentReady = h<VkSemaphore>(27),
        .graphicsFence = h<VkFence>(28), .commandPoolFamily = 0, .submitQueueFamily = 0,
        .submitQueueFlags = VK_QUEUE_GRAPHICS_BIT, .singleSwapchain = true, .fifo = true,
        .hiddenBlitDestinationSupported = true, .maintenanceReleaseCapable = true,
        .presentFences = {h<VkFence>(29), h<VkFence>(30)}, .state = &state};
    uint32_t acquired = 0;
    path.acquire = [&](VkSemaphore) {
        ++acquired;
        return D3B2HiddenImage{h<VkImage>(30 + acquired), acquired, {1, 1}};
    };
    path.record = [&](const auto&, const auto&) { events.push_back(1); };
    path.submit = [&] { events.push_back(2); return VK_SUCCESS; };
    path.present = [&](const auto&, VkSemaphore) { events.push_back(3); return VK_SUCCESS; };
    path.presentWithFence = [&](const auto&, VkSemaphore, VkFence fence) {
        assert(fence == path.presentFences.generated || fence == path.presentFences.original);
        events.push_back(4); return VK_SUCCESS;
    };
    path.waitGraphicsFence = [&] { return true; };
    path.generatedIntegrity = [&] { return true; };
    path.originalIdentity = [&] { return true; };
    path.retire = [&] {};
    path.emitMarker = [&] {};
    path.releaseAcquiredImages = [&](const std::vector<uint32_t>&) { return VK_SUCCESS; };
    assert(executeReusableTerminal(path, true) == VK_SUCCESS);
    assert(events == std::vector<int>({1, 2, 4, 4}));

    assert(evaluateD3B3Retirement({true, true, true, true, true, true, true, true, true})
        == D3B3RetirementStatus::RETIRED);
    assert(evaluateD3B3Retirement({true, true, false, true, true, true, true, true, true})
        == D3B3RetirementStatus::TIMEOUT);
    assert(shouldStopWorker(D3B3ShutdownPolicy::IMMEDIATE_ONE_SHOT, false));
    assert(!shouldStopWorker(D3B3ShutdownPolicy::DEFER_UNTIL_FINITE_STOP, false));
    assert(shouldStopWorker(D3B3ShutdownPolicy::DEFER_UNTIL_FINITE_STOP, true));
}
