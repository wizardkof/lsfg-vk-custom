/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "acquire_fence_proxy_registry.hpp"

namespace lsfgvk::layer {

std::optional<AcquireFenceProxyRegistry::Prepared>
AcquireFenceProxyRegistry::prepare(VkFence fence, VkFence proxy,
        std::shared_ptr<void> backing) noexcept {
    if (!lifecycles || proxy == VK_NULL_HANDLE || !backing) return std::nullopt;
    const auto identity = lifecycles->identity(fence);
    if (!identity.valid()) return std::nullopt;
    try {
        const std::scoped_lock lock(mutex);
        VkFence previousProxy{};
        std::shared_ptr<void> previousBacking;
        BorrowedFenceLifecycleIdentity previousIdentity;
        if (const auto old = associations.find(fence); old != associations.end()) {
            previousProxy = old->second.proxy;
            previousBacking = old->second.backing;
            previousIdentity = old->second.identity;
        }
        associations.insert_or_assign(fence,
            Association{identity, VK_NULL_HANDLE, {}});
        return Prepared{identity, proxy, std::move(backing), previousProxy,
            std::move(previousBacking), previousIdentity};
    } catch (...) { return std::nullopt; }
}

void AcquireFenceProxyRegistry::commit(Prepared& prepared) noexcept {
    if (!prepared.valid()) return;
    const std::scoped_lock lock(mutex);
    const auto current = lifecycles->identity(prepared.identity.fence);
    if (current.fenceObjectId != prepared.identity.fenceObjectId
            || current.fenceLifecycleEpoch != prepared.identity.fenceLifecycleEpoch)
        return;
    auto it = associations.find(prepared.identity.fence);
    if (it == associations.end()) return;
    it->second = {prepared.identity, prepared.proxy, std::move(prepared.backing)};
    prepared = {};
}

void AcquireFenceProxyRegistry::abort(Prepared& prepared) noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = associations.find(prepared.identity.fence);
    if (it != associations.end()
            && it->second.identity.fenceObjectId == prepared.identity.fenceObjectId
            && it->second.proxy == VK_NULL_HANDLE) {
        if (prepared.previousIdentity.valid())
            it->second = {prepared.previousIdentity, prepared.previousProxy,
                std::move(prepared.previousBacking)};
        else
            associations.erase(it);
    }
    prepared = {};
}

bool AcquireFenceProxyRegistry::currentLocked(const Association& value) const noexcept {
    const auto current = lifecycles->identity(value.identity.fence);
    return current.fenceObjectId == value.identity.fenceObjectId
        && current.fenceLifecycleEpoch == value.identity.fenceLifecycleEpoch;
}

VkFence AcquireFenceProxyRegistry::project(VkFence fence) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = associations.find(fence);
    return it != associations.end() && currentLocked(it->second)
        ? it->second.proxy : fence;
}

std::vector<VkFence> AcquireFenceProxyRegistry::project(
        uint32_t count, const VkFence* fences) const {
    std::vector<VkFence> result;
    result.reserve(count);
    const std::scoped_lock lock(mutex);
    for (uint32_t i = 0; i < count; ++i) {
        const auto it = associations.find(fences[i]);
        result.push_back(it != associations.end() && currentLocked(it->second)
            ? it->second.proxy : fences[i]);
    }
    return result;
}

void AcquireFenceProxyRegistry::resetSuccess(
        uint32_t count, const VkFence* fences) noexcept {
    const std::scoped_lock lock(mutex);
    for (uint32_t i = 0; fences && i < count; ++i) associations.erase(fences[i]);
}

void AcquireFenceProxyRegistry::destroy(VkFence fence) noexcept {
    const std::scoped_lock lock(mutex);
    associations.erase(fence);
}

void AcquireFenceProxyRegistry::importSuccess(VkFence fence) noexcept {
    destroy(fence);
}

size_t AcquireFenceProxyRegistry::associationCount() const noexcept {
    const std::scoped_lock lock(mutex);
    return associations.size();
}

bool AcquireFenceProxyRegistry::applicationOperationAllowed(
        VkFence fence) const noexcept {
    const std::scoped_lock lock(mutex);
    const auto it = associations.find(fence);
    return it == associations.end() || it->second.proxy != VK_NULL_HANDLE;
}

}
