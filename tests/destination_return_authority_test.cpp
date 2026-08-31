/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/destination_return_authority.hpp"
#include "lsfg-vk-common/helpers/owned_fd.hpp"

#include <array>
#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <vector>
#include <unistd.h>

namespace {
template <typename T> T handle(uintptr_t value) { return reinterpret_cast<T>(value); }
}

int main() {
    // DRA-F1/F2/F3: exact cardinality, including the valid zero/zero case.
    assert(!vk::destinationReturnCardinalityValid(1, 0));
    assert(!vk::destinationReturnCardinalityValid(2, 1));
    assert(vk::destinationReturnCardinalityValid(1, 1));
    assert(vk::destinationReturnCardinalityValid(4, 4));
    assert(vk::destinationReturnCardinalityValid(0, 0));

    // An early cardinality rejection leaves all transferred descriptors under
    // RAII ownership; none can leak or be double-consumed.
    int pipeFds[2]{};
    assert(::pipe(pipeFds) == 0);
    const int observedFd = pipeFds[0];
    {
        std::vector<ls::OwnedFd> owned;
        owned.emplace_back(pipeFds[0]);
        owned.emplace_back(pipeFds[1]);
        assert(!vk::destinationReturnCardinalityValid(1, owned.size()));
    }
    errno = 0;
    assert(::fcntl(observedFd, F_GETFD) == -1 && errno == EBADF);

    // DRA-4/8/9/12: first use has no wait; accepted cycles commit exact values.
    vk::DestinationReturnBackendState backend;
    vk::DestinationReturnLayerState layer;
    assert(!backend.requiresReturnWait());
    assert(backend.pendingWriteGeneration() == 1);
    backend.backendWriteAccepted();
    assert(backend.requiresReturnWait());
    assert(backend.expectedReturnGeneration() == 1);
    assert(layer.pendingReturnGeneration() == 1);
    layer.returnSubmitAccepted();
    backend.backendWriteAccepted();
    assert(backend.expectedReturnGeneration() == 2);
    assert(layer.pendingReturnGeneration() == 2);
    layer.returnSubmitAccepted();

    // DRA-10/11/13: independent destinations and four-destination capacity.
    std::array<vk::DestinationReturnBackendState, 4> backends{};
    std::array<vk::DestinationReturnLayerState, 4> layers{};
    for (std::size_t i = 0; i < backends.size(); ++i) {
        backends[i].backendWriteAccepted();
        layers[i].returnSubmitAccepted();
    }
    backends[0].backendWriteAccepted();
    layers[0].returnSubmitAccepted();
    assert(backends[0].expectedReturnGeneration() == 2);
    assert(layers[0].pendingReturnGeneration() == 3);
    for (std::size_t i = 1; i < backends.size(); ++i) {
        assert(backends[i].expectedReturnGeneration() == 1);
        assert(layers[i].pendingReturnGeneration() == 2);
    }

    // DRA-14/15/16/17/18: pre-enqueue failures and device loss do not
    // fabricate a commit; the state changes only at accepted-submit methods.
    vk::DestinationReturnBackendState backendRejected;
    vk::DestinationReturnLayerState layerRejected;
    assert(backendRejected.firstUse && backendRejected.nextGeneration == 1);
    assert(layerRejected.nextGeneration == 1);

    // DRA-19/20: replacement lifetime starts with a distinct state authority;
    // no old generation value is imported into the replacement object.
    vk::DestinationReturnBackendState oldLifetime;
    oldLifetime.backendWriteAccepted();
    vk::DestinationReturnBackendState replacementLifetime;
    assert(oldLifetime.expectedReturnGeneration() == 1);
    assert(!replacementLifetime.requiresReturnWait());
    assert(replacementLifetime.pendingWriteGeneration() == 1);

    // DRA-F4 backend production helper: first use has only prepass; second
    // use carries the exact destination return generation.
    const auto prepass = handle<VkSemaphore>(0x100);
    const auto destinationReturn = handle<VkSemaphore>(0x101);
    const auto sync = handle<VkSemaphore>(0x102);
    vk::BackendDestinationSubmitStorage backendSubmit;
    vk::DestinationReturnBackendState observedBackend;
    vk::prepareBackendDestinationSubmit(prepass, 11, destinationReturn,
        observedBackend, sync, 12, backendSubmit);
    assert(backendSubmit.submission.timelineWaits.size() == 1);
    assert(backendSubmit.submission.timelineWaits[0].semaphore == prepass);
    assert(backendSubmit.submission.timelineWaits[0].value == 11);
    observedBackend.backendWriteAccepted();
    vk::prepareBackendDestinationSubmit(prepass, 21, destinationReturn,
        observedBackend, sync, 22, backendSubmit);
    assert(backendSubmit.submission.timelineWaits.size() == 2);
    assert(backendSubmit.submission.timelineWaits[1].semaphore == destinationReturn);
    assert(backendSubmit.submission.timelineWaits[1].value == 1);
    assert(backendSubmit.submission.timelineSignals[0].semaphore == sync);
    assert(backendSubmit.submission.timelineSignals[0].value == 22);

    const auto initialAcquire = vk::destinationInitialAcquireFromLayer(
        handle<VkImage>(0x200), 7);
    const auto reuseAcquire = vk::destinationAcquireFromLayer(
        handle<VkImage>(0x200), 7);
    assert(initialAcquire.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
    assert(initialAcquire.srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);
    assert(reuseAcquire.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    assert(reuseAcquire.srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);

    // DRA-F4 layer production helper: exact ready wait and generation return
    // signal, while the local binary chain remains separate.
    const std::array<VkSemaphore, 2> binaryWaits{
        handle<VkSemaphore>(0x300), handle<VkSemaphore>(0x301)};
    const std::array<VkSemaphore, 2> binarySignals{
        handle<VkSemaphore>(0x302), handle<VkSemaphore>(0x303)};
    vk::LayerDestinationReturnSubmitStorage layerSubmit;
    vk::DestinationReturnLayerState observedLayer;
    vk::prepareLayerDestinationReturnSubmit(binaryWaits, binarySignals,
        sync, 32, destinationReturn, observedLayer, layerSubmit);
    assert(layerSubmit.submission.timelineWaits.size() == 1);
    assert(layerSubmit.submission.timelineWaits[0].semaphore == sync);
    assert(layerSubmit.submission.timelineWaits[0].value == 32);
    assert(layerSubmit.submission.timelineSignals.size() == 1);
    assert(layerSubmit.submission.timelineSignals[0].semaphore == destinationReturn);
    assert(layerSubmit.submission.timelineSignals[0].value == 1);
    assert(layerSubmit.submission.binarySignals[1] == binarySignals[1]);
    observedLayer.returnSubmitAccepted();
    vk::prepareLayerDestinationReturnSubmit(binaryWaits, binarySignals,
        sync, 42, destinationReturn, observedLayer, layerSubmit);
    assert(layerSubmit.submission.timelineSignals[0].value == 2);

    const auto release = vk::destinationReleaseToBackend(
        handle<VkImage>(0x200), 7);
    assert(release.dstQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);
    assert(release.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        && release.newLayout == VK_IMAGE_LAYOUT_GENERAL);
}
