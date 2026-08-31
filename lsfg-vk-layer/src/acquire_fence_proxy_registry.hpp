/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "borrowed_present_fence_registry.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lsfgvk::layer {

class AcquireFenceProxyRegistry final {
public:
    struct Prepared {
        BorrowedFenceLifecycleIdentity identity;
        VkFence proxy{VK_NULL_HANDLE};
        std::shared_ptr<void> backing;
        VkFence previousProxy{VK_NULL_HANDLE};
        std::shared_ptr<void> previousBacking;
        BorrowedFenceLifecycleIdentity previousIdentity;
        [[nodiscard]] bool valid() const noexcept {
            return identity.valid() && proxy != VK_NULL_HANDLE && backing;
        }
    };

    explicit AcquireFenceProxyRegistry(
        std::shared_ptr<BorrowedPresentFenceRegistry> lifecycles) noexcept
        : lifecycles(std::move(lifecycles)) {}

    [[nodiscard]] std::optional<Prepared> prepare(VkFence, VkFence,
        std::shared_ptr<void>) noexcept;
    void commit(Prepared&) noexcept;
    void abort(Prepared&) noexcept;
    [[nodiscard]] VkFence project(VkFence) const noexcept;
    [[nodiscard]] std::vector<VkFence> project(
        uint32_t, const VkFence*) const;
    void resetSuccess(uint32_t, const VkFence*) noexcept;
    void destroy(VkFence) noexcept;
    void importSuccess(VkFence) noexcept;
    [[nodiscard]] size_t associationCount() const noexcept;
    [[nodiscard]] bool applicationOperationAllowed(VkFence) const noexcept;

private:
    struct Association {
        BorrowedFenceLifecycleIdentity identity;
        VkFence proxy{};
        std::shared_ptr<void> backing;
    };
    [[nodiscard]] bool currentLocked(const Association&) const noexcept;
    std::shared_ptr<BorrowedPresentFenceRegistry> lifecycles;
    mutable std::mutex mutex;
    std::unordered_map<VkFence, Association> associations;
};

}
