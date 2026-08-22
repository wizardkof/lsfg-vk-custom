#include "d3b1_present_path.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <string>
#include <vector>

using namespace lsfgvk::layer;

namespace {
template<class T> T h(uintptr_t value) { return reinterpret_cast<T>(value); }
struct Trace {
    std::vector<std::string> events;
    std::vector<vk::Barrier> pre, post;
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    VkImage source{}, destination{};
    VkExtent2D sourceExtent{}, destinationExtent{};
    VkSemaphore signal{}, presentWait{};
    VkFence fence{};
    VkResult presentResult{VK_SUCCESS};
    bool fenceResult{true}, diagnosticsResult{true};
    int acquires{}, submits{}, presents{}, fenceWaits{}, retires{}, markers{};
};

D3B1PresentPath makePath(Trace& t, D3B1PresentationState& state,
        bool differentFamily = true) {
    D3B1PresentPath path{
        .source = {.image = h<VkImage>(0x101),
            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {500, 500},
            .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .modifier = 0x55,
            .sourceQueueFamily = differentFamily ? 3U : 5U,
            .destinationQueueFamily = differentFamily ? 7U : 5U,
            .ownershipAcquireRequired = differentFamily},
        .hiddenAcquire = h<VkSemaphore>(0x201),
        .returnedForPresent = h<VkSemaphore>(0x202),
        .finalPresentSemaphore = h<VkSemaphore>(0x203),
        .renderFence = h<VkFence>(0x301),
        .commandPoolFamily = differentFamily ? 7U : 5U,
        .submitQueueFamily = differentFamily ? 7U : 5U,
        .submitQueueFlags = VK_QUEUE_GRAPHICS_BIT,
        .sourceBlitSupported = true, .destinationBlitSupported = true,
        .state = &state};
    path.acquireHidden = [&] {
        t.events.emplace_back("HIDDEN_ACQUIRE"); ++t.acquires;
        return D3B1HiddenImage{h<VkImage>(0x102), 4, {640, 480}};
    };
    path.recordBlit = [&](const auto& pre, VkImage source, VkImage destination,
            VkExtent2D sourceExtent, VkExtent2D destinationExtent, const auto& post) {
        t.events.emplace_back("RECORD_BLIT"); t.pre = pre; t.post = post;
        t.source = source; t.destination = destination;
        t.sourceExtent = sourceExtent; t.destinationExtent = destinationExtent;
    };
    path.submitAndPresent = [&](const auto& waits, const auto& stages,
            VkSemaphore signal, VkFence fence, uint32_t index) {
        t.events.emplace_back("OFFLOAD_SUBMIT"); ++t.submits;
        t.waits = waits; t.stages = stages; t.signal = signal; t.fence = fence;
        assert(index == 4); t.events.emplace_back("REAL_QUEUE_PRESENT");
        ++t.presents; t.presentWait = signal;
        return D3B1SubmitPresentResult{t.presentResult, true};
    };
    path.waitRenderFence = [&] {
        t.events.emplace_back("RENDER_FENCE_WAIT"); ++t.fenceWaits;
        return t.fenceResult;
    };
    path.completeDiagnostics = [&] {
        t.events.emplace_back("A_B_FNV_VALIDATE"); return t.diagnosticsResult;
    };
    path.retire = [&] { t.events.emplace_back("RETIRE"); ++t.retires; };
    path.emitMarker = [&] { t.events.emplace_back("MARKER"); ++t.markers; };
    return path;
}
size_t event(const Trace& t, const std::string& value) {
    return std::ranges::find(t.events, value) - t.events.begin();
}
bool fails(D3B1PresentPath& path) {
    try { static_cast<void>(executeD3B1PresentPath(path)); }
    catch (...) { return true; }
    return false;
}
}

int main() {
    assert(d3b1OneShotRequested("1"));
    assert(!d3b1OneShotRequested(nullptr));
    assert(!d3b1OneShotRequested("0"));
    assert(!d3b1OneShotRequested("true"));
    assert(d3b1StopAfterTerminalPass(true, D3B1PresentationState::PASS));
    assert(!d3b1StopAfterTerminalPass(false, D3B1PresentationState::PASS));
    assert(!d3b1StopAfterTerminalPass(true, D3B1PresentationState::PRESENT_SUCCESS));
    assert(d3b1PresentEligible(true, true, false, true, true,
        D3B1PresentationState::IDLE));
    assert(!d3b1PresentEligible(false, true, false, true, true,
        D3B1PresentationState::IDLE));
    assert(!d3b1PresentEligible(true, false, false, true, true,
        D3B1PresentationState::IDLE));
    assert(!d3b1PresentEligible(true, true, true, true, true,
        D3B1PresentationState::IDLE));
    assert(!d3b1PresentEligible(true, true, false, true, true,
        D3B1PresentationState::FAILED));
    {
        Trace t; auto state = D3B1PresentationState::A_HANDOFF_SUBMITTED;
        auto path = makePath(t, state);
        assert(executeD3B1PresentPath(path) == VK_SUCCESS);
        assert(state == D3B1PresentationState::PASS);
        assert(t.acquires == 1 && t.submits == 1 && t.presents == 1);
        assert(t.fenceWaits == 1 && t.retires == 1 && t.markers == 1);
        assert(t.pre.size() == 2 && t.post.size() == 1);
        const auto& acquire = t.pre.front();
        assert(acquire.image == path.source.image);
        assert(acquire.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            && acquire.newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        assert(acquire.srcQueueFamilyIndex == 3 && acquire.dstQueueFamilyIndex == 7);
        assert(acquire.srcAccessMask == 0
            && acquire.dstAccessMask == VK_ACCESS_TRANSFER_READ_BIT);
        assert(t.source == path.source.image && t.destination == h<VkImage>(0x102));
        assert(t.sourceExtent.width == 500 && t.sourceExtent.height == 500);
        assert(t.destinationExtent.width == 640 && t.destinationExtent.height == 480);
        assert(t.waits == std::vector<VkSemaphore>({path.hiddenAcquire,
            path.returnedForPresent}));
        assert(t.stages.size() == t.waits.size()
            && std::ranges::all_of(t.stages, [](auto stage) {
                return stage == VK_PIPELINE_STAGE_TRANSFER_BIT;
            }));
        assert(t.signal == path.finalPresentSemaphore
            && t.presentWait == path.finalPresentSemaphore && t.fence == path.renderFence);
        assert(event(t, "OFFLOAD_SUBMIT") < event(t, "REAL_QUEUE_PRESENT"));
        assert(event(t, "REAL_QUEUE_PRESENT") < event(t, "RENDER_FENCE_WAIT"));
        assert(event(t, "RENDER_FENCE_WAIT") < event(t, "A_B_FNV_VALIDATE"));
        assert(event(t, "A_B_FNV_VALIDATE") < event(t, "RETIRE"));
        assert(event(t, "RETIRE") < event(t, "MARKER"));
    }
    {
        Trace t; auto state = D3B1PresentationState::A_HANDOFF_SUBMITTED;
        auto path = makePath(t, state, false);
        assert(executeD3B1PresentPath(path) == VK_SUCCESS);
        assert(t.pre.size() == 1 && t.waits.at(1) == path.returnedForPresent);
    }
    for (int variant = 0; variant < 9; ++variant) {
        Trace t; auto state = D3B1PresentationState::A_HANDOFF_SUBMITTED;
        auto path = makePath(t, state);
        if (variant == 0) path.sourceBlitSupported = false;
        if (variant == 1) path.destinationBlitSupported = false;
        if (variant == 2) path.submitQueueFamily = 9;
        if (variant == 3) t.presentResult = VK_SUBOPTIMAL_KHR;
        if (variant == 4) t.presentResult = VK_ERROR_OUT_OF_DATE_KHR;
        if (variant == 5) t.presentResult = VK_ERROR_DEVICE_LOST;
        if (variant == 6) t.fenceResult = false;
        if (variant == 7) t.diagnosticsResult = false;
        if (variant == 8) path.submitQueueFlags = VK_QUEUE_COMPUTE_BIT;
        assert(fails(path));
        assert(state == D3B1PresentationState::FAILED && t.markers == 0);
        assert(t.retires == 1);
        const int priorPresents = t.presents;
        assert(fails(path));
        assert(t.presents == priorPresents && t.markers == 0);
    }
    return 0;
}
