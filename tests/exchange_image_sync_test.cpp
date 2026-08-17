/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/vulkan/exchange_image_sync.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

namespace {
    constexpr uint32_t LAYER_FAMILY = 3;
    constexpr uint32_t BACKEND_FAMILY = 7;

    void expectMatching(const VkImageMemoryBarrier& release,
            const VkImageMemoryBarrier& acquire,
            uint32_t releaseFamily, uint32_t acquireFamily,
            VkAccessFlags releaseAccess, VkAccessFlags acquireAccess,
            VkImageLayout oldLayout, VkImageLayout newLayout) {
        assert(release.srcQueueFamilyIndex == releaseFamily);
        assert(release.dstQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);
        assert(acquire.srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);
        assert(acquire.dstQueueFamilyIndex == acquireFamily);
        assert(release.oldLayout == acquire.oldLayout);
        assert(release.newLayout == acquire.newLayout);
        assert(release.oldLayout == oldLayout);
        assert(release.newLayout == newLayout);
        assert(release.srcAccessMask == releaseAccess);
        assert(release.dstAccessMask == 0);
        assert(acquire.srcAccessMask == 0);
        assert(acquire.dstAccessMask == acquireAccess);
        assert(release.subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
        assert(release.subresourceRange.baseMipLevel == 0);
        assert(release.subresourceRange.levelCount == 1);
        assert(release.subresourceRange.baseArrayLayer == 0);
        assert(release.subresourceRange.layerCount == 1);
        assert(release.subresourceRange.aspectMask == acquire.subresourceRange.aspectMask);
        assert(release.subresourceRange.baseMipLevel == acquire.subresourceRange.baseMipLevel);
        assert(release.subresourceRange.levelCount == acquire.subresourceRange.levelCount);
        assert(release.subresourceRange.baseArrayLayer == acquire.subresourceRange.baseArrayLayer);
        assert(release.subresourceRange.layerCount == acquire.subresourceRange.layerCount);
    }

    void testBarrierContracts() {
        const VkImage image{};
        expectMatching(
            vk::sourceReleaseToBackend(image, LAYER_FAMILY),
            vk::sourceAcquireFromLayer(image, BACKEND_FAMILY),
            LAYER_FAMILY, BACKEND_FAMILY,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        expectMatching(
            vk::sourceReleaseToLayer(image, BACKEND_FAMILY),
            vk::sourceAcquireFromBackend(image, LAYER_FAMILY),
            BACKEND_FAMILY, LAYER_FAMILY,
            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        expectMatching(
            vk::destinationInitialReleaseToBackend(image, LAYER_FAMILY),
            vk::destinationInitialAcquireFromLayer(image, BACKEND_FAMILY),
            LAYER_FAMILY, BACKEND_FAMILY, 0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        expectMatching(
            vk::destinationReleaseToLayer(image, BACKEND_FAMILY),
            vk::destinationAcquireFromBackend(image, LAYER_FAMILY),
            BACKEND_FAMILY, LAYER_FAMILY,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        expectMatching(
            vk::destinationReleaseToBackend(image, LAYER_FAMILY),
            vk::destinationAcquireFromLayer(image, BACKEND_FAMILY),
            LAYER_FAMILY, BACKEND_FAMILY,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        const auto ordinary = vk::makeImageBarrier(image,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        assert(ordinary.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        assert(ordinary.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    }

    enum class Owner {
        Layer,
        ReleasedToBackend,
        Backend,
        ReleasedToLayer
    };

    struct ResourceState {
        Owner owner{Owner::Layer};

        bool releaseToBackend() {
            if (owner != Owner::Layer)
                return false;
            owner = Owner::ReleasedToBackend;
            return true;
        }
        bool acquireBackend() {
            if (owner != Owner::ReleasedToBackend)
                return false;
            owner = Owner::Backend;
            return true;
        }
        bool releaseToLayer() {
            if (owner != Owner::Backend)
                return false;
            owner = Owner::ReleasedToLayer;
            return true;
        }
        bool acquireLayer() {
            if (owner != Owner::ReleasedToLayer)
                return false;
            owner = Owner::Layer;
            return true;
        }
    };

    void testFirstUseAndReuse() {
        // Both sources are produced by A on the first real frame. Thereafter,
        // each alternating resource follows acquire -> write -> release.
        std::array<ResourceState, 2> sources{};
        for (auto& source : sources) {
            assert(source.owner == Owner::Layer); // UNDEFINED -> TRANSFER_DST + write
            assert(source.releaseToBackend());
            assert(source.acquireBackend());
        }
        assert(sources[1].releaseToLayer()); // previous resource after last Generate
        assert(sources[1].acquireLayer());
        assert(!sources[1].acquireLayer()); // no duplicate acquire / double local ownership
        assert(sources[1].releaseToBackend());
        assert(sources[1].acquireBackend());

        ResourceState destination{};
        assert(!destination.acquireBackend()); // acquire requires A's initial release
        assert(destination.releaseToBackend()); // UNDEFINED -> GENERAL
        assert(destination.acquireBackend());
        assert(!destination.acquireBackend());
        assert(destination.releaseToLayer());   // storage write -> transfer read
        assert(destination.acquireLayer());
        assert(destination.releaseToBackend()); // transfer read -> next storage write
        assert(destination.acquireBackend());

        // Resource-specific alternation across g=0/g>0 transitions. The
        // previous source is returned every frame, while each destination is
        // acquired only when that pass is actually scheduled.
        std::array<ResourceState, 2> alternatingSources{};
        std::array<ResourceState, 7> destinations{};
        for (auto& source : alternatingSources) {
            assert(source.releaseToBackend());
            assert(source.acquireBackend());
        }
        for (auto& output : destinations)
            assert(output.releaseToBackend()); // one-time initial handoff

        const std::array<size_t, 6> generations{ 0, 4, 0, 1, 7, 0 };
        for (size_t frame = 0; frame < generations.size(); ++frame) {
            const size_t current = frame % 2;
            const size_t returned = (frame + 1) % 2;
            if (frame > 0) {
                assert(alternatingSources[current].acquireLayer());
                assert(alternatingSources[current].releaseToBackend());
                assert(alternatingSources[current].acquireBackend());
            }
            assert(alternatingSources[returned].releaseToLayer());

            for (size_t i = 0; i < generations[frame]; ++i) {
                assert(destinations[i].acquireBackend());
                assert(destinations[i].releaseToLayer());
                assert(destinations[i].acquireLayer());
                assert(destinations[i].releaseToBackend());
            }
        }
    }

    void validateTimelineSequence(const std::vector<size_t>& generations) {
        uint64_t base = 1;
        uint64_t currentCounter = 0;
        std::set<uint64_t> produced;
        std::array<std::optional<uint64_t>, 2> sourceReturns{};

        for (size_t frame = 0; frame < generations.size(); ++frame) {
            const size_t count = generations[frame];
            const auto layerFrame = vk::makeExchangeTimelineFrame(base, count);
            const auto backendFrame = vk::makeExchangeTimelineFrame(base, count);
            assert(layerFrame.nextBase == backendFrame.nextBase);
            const auto currentSourceReturn = sourceReturns[frame % 2];
            if (currentSourceReturn.has_value()) {
                assert(produced.contains(*currentSourceReturn));
                assert(*currentSourceReturn < layerFrame.sourceReady);
            }
            assert(layerFrame.sourceReady > currentCounter);
            assert(produced.insert(layerFrame.sourceReady).second);
            currentCounter = layerFrame.sourceReady;

            if (count == 0) {
                assert(layerFrame.sourceReturn == layerFrame.sourceReady + 1);
                assert(produced.insert(layerFrame.sourceReturn).second);
                assert(layerFrame.sourceReturn > currentCounter);
                currentCounter = layerFrame.sourceReturn;
            } else {
                for (size_t i = 0; i < count; ++i) {
                    const uint64_t ready = layerFrame.destinationReady(i);
                    assert(ready > currentCounter);
                    assert(produced.insert(ready).second);
                    currentCounter = ready;
                }
                assert(layerFrame.sourceReturn
                    == layerFrame.destinationReady(count - 1));
            }

            assert(produced.contains(layerFrame.sourceReady));
            assert(produced.contains(layerFrame.sourceReturn));
            assert(layerFrame.nextBase == currentCounter + 1);
            sourceReturns[(frame + 1) % 2] = layerFrame.sourceReturn;
            base = layerFrame.nextBase;
        }
    }

    void testTimelineSchedules() {
        validateTimelineSequence({ 0 });
        validateTimelineSequence({ 1 });
        validateTimelineSequence({ 2 });
        validateTimelineSequence({ 4 });
        validateTimelineSequence({ 7 }); // Fixed capacity
        validateTimelineSequence({ 0, 0, 0 });
        validateTimelineSequence({ 0, 4, 0 });
        validateTimelineSequence({ 4, 0, 4 });
        validateTimelineSequence({ 0, 1, 2, 4, 7, 0 });
    }
}

int main() {
    testBarrierContracts();
    testFirstUseAndReuse();
    testTimelineSchedules();
}
