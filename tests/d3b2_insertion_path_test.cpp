#include "d3b2_insertion_path.hpp"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
using namespace lsfgvk::layer;
template<class T> T h(uintptr_t n) { return reinterpret_cast<T>(n); }
int main() {
    assert(!d3b2InsertionRequested(nullptr));
    assert(!d3b2InsertionRequested("0"));
    assert(!d3b2InsertionRequested("true"));
    assert(d3b2InsertionRequested("1"));
    std::vector<std::string> events;
    D3B2InsertionState state = D3B2InsertionState::INACTIVE;
    int acquires = 0, submits = 0, presents = 0, waits = 0, markers = 0;
    D3B2InsertionPath path{
        .generated = {.image = h<VkImage>(1), .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {500, 500}, .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .sourceQueueFamily = 1, .destinationQueueFamily = 0,
            .ownershipAcquireRequired = true, .blitSourceSupported = true},
        .original = {.image = h<VkImage>(2), .format = VK_FORMAT_B8G8R8A8_UNORM,
            .extent = {500, 500}, .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .sourceQueueFamily = 0, .destinationQueueFamily = 0,
            .blitSourceSupported = true},
        .originalReady = h<VkSemaphore>(3), .returnedForGraphics = h<VkSemaphore>(4),
        .acquireGenerated = h<VkSemaphore>(5), .acquireOriginal = h<VkSemaphore>(6),
        .generatedPresentReady = h<VkSemaphore>(7), .originalPresentReady = h<VkSemaphore>(8),
        .graphicsFence = h<VkFence>(9), .commandPoolFamily = 0, .submitQueueFamily = 0,
        .submitQueueFlags = VK_QUEUE_GRAPHICS_BIT, .singleSwapchain = true, .fifo = true,
        .hiddenBlitDestinationSupported = true, .state = &state};
    path.acquire = [&](VkSemaphore sem) {
        events.emplace_back(sem == path.acquireGenerated ? "ACQUIRE_G" : "ACQUIRE_O");
        ++acquires;
        return D3B2HiddenImage{h<VkImage>(10 + acquires), static_cast<uint32_t>(acquires), {500, 500}};
    };
    path.record = [&](const auto&, const auto&) { events.emplace_back("BLIT_G_THEN_BLIT_O"); };
    path.submit = [&] { events.emplace_back("SUBMIT"); ++submits; return VK_SUCCESS; };
    path.present = [&](const auto&, VkSemaphore sem) {
        events.emplace_back(sem == path.generatedPresentReady ? "PRESENT_G" : "PRESENT_O");
        ++presents; return VK_SUCCESS;
    };
    path.waitGraphicsFence = [&] { events.emplace_back("FENCE"); ++waits; return true; };
    path.generatedIntegrity = [&] { return true; };
    path.originalIdentity = [&] { return true; };
    path.retire = [&] { events.emplace_back("RETIRE"); };
    path.emitMarker = [&] { ++markers; events.emplace_back("D3B2_MARKER"); };
    path.maintenanceReleaseCapable = true;
    path.releaseAcquiredImages = [&](const std::vector<uint32_t>&) {
        events.emplace_back("RELEASE");
        return VK_SUCCESS;
    };
    assert(executeD3B2Insertion(path) == VK_SUCCESS);
    assert(state == D3B2InsertionState::PASS);
    assert(acquires == 2 && submits == 1 && presents == 2 && waits == 1 && markers == 1);
    assert(events[0] == "ACQUIRE_G" && events[1] == "ACQUIRE_O");
    assert(events[2] == "BLIT_G_THEN_BLIT_O" && events[3] == "SUBMIT");
    assert(events[4] == "PRESENT_G" && events[5] == "PRESENT_O");
    assert(events[6] == "FENCE" && events[7] == "RETIRE" && events[8] == "D3B2_MARKER");
    bool reentered = false;
    try { static_cast<void>(executeD3B2Insertion(path)); } catch (...) { reentered = true; }
    assert(reentered && presents == 2 && markers == 1);

    // Failure-closure matrix: every acquired image is either handed to
    // present or released exactly once through the maintenance1 seam.
    enum class Failure { AcquireOriginal, Record, Submit, PresentGenerated,
        PresentOriginal, Fence, DeviceLostSubmit };
    for (const auto failure : {Failure::AcquireOriginal, Failure::Record,
            Failure::Submit, Failure::PresentGenerated, Failure::PresentOriginal,
            Failure::Fence, Failure::DeviceLostSubmit}) {
        state = D3B2InsertionState::INACTIVE;
        int releaseCalls = 0, releaseImages = 0, fenceCalls = 0;
        int acquireCount = 0;
        path.acquire = [&](VkSemaphore sem) {
            ++acquireCount;
            if (failure == Failure::AcquireOriginal && sem == path.acquireOriginal)
                throw std::runtime_error("acquire O");
            return D3B2HiddenImage{h<VkImage>(20 + acquireCount),
                static_cast<uint32_t>(acquireCount), {500, 500}};
        };
        path.record = [&] (const auto&, const auto&) {
            if (failure == Failure::Record) throw std::runtime_error("record");
        };
        path.submit = [&] {
            if (failure == Failure::Submit) return VK_ERROR_INITIALIZATION_FAILED;
            if (failure == Failure::DeviceLostSubmit) return VK_ERROR_DEVICE_LOST;
            return VK_SUCCESS;
        };
        path.present = [&] (const auto&, VkSemaphore sem) {
            if (failure == Failure::PresentGenerated && sem == path.generatedPresentReady)
                return VK_ERROR_OUT_OF_DATE_KHR;
            if (failure == Failure::PresentOriginal && sem == path.originalPresentReady)
                return VK_ERROR_OUT_OF_DATE_KHR;
            return VK_SUCCESS;
        };
        path.waitGraphicsFence = [&] {
            ++fenceCalls;
            return failure != Failure::Fence;
        };
        path.releaseAcquiredImages = [&] (const std::vector<uint32_t>& indices) {
            ++releaseCalls; releaseImages += static_cast<int>(indices.size());
            return VK_SUCCESS;
        };
        bool threw = false;
        try { static_cast<void>(executeD3B2Insertion(path)); } catch (...) { threw = true; }
        assert(threw && state == D3B2InsertionState::FAILED);
        const bool deviceLost = failure == Failure::DeviceLostSubmit;
        if (deviceLost || failure == Failure::Fence)
            assert(releaseCalls == 0);
        else if (failure == Failure::AcquireOriginal)
            assert(releaseCalls == 1 && releaseImages == 1);
        else if (failure == Failure::PresentOriginal)
            assert(releaseCalls == 0);
        else if (failure == Failure::PresentGenerated)
            assert(releaseCalls == 1 && releaseImages == 1);
        else
            assert(releaseCalls == 1 && releaseImages == 2);
    }
    return 0;
}
