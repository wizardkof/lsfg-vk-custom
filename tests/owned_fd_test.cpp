/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-common/helpers/owned_fd.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"

#include <cassert>
#include <cerrno>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

namespace {
    std::pair<ls::OwnedFd, ls::OwnedFd> makePipe() {
        int descriptors[2]{ -1, -1 };
        assert(::pipe(descriptors) == 0);
        return { ls::OwnedFd(descriptors[0]), ls::OwnedFd(descriptors[1]) };
    }

    bool isOpen(int fd) {
        errno = 0;
        const auto result = ::fcntl(fd, F_GETFD);
        return result != -1 || errno != EBADF;
    }
}

int main() {
    static_assert(!std::is_copy_constructible_v<ls::OwnedFd>);
    static_assert(!std::is_copy_assignable_v<ls::OwnedFd>);
    static_assert(std::is_nothrow_move_constructible_v<ls::OwnedFd>);
    static_assert(std::is_nothrow_move_assignable_v<ls::OwnedFd>);
    static_assert(!std::is_copy_constructible_v<vk::ExternalImage>);
    static_assert(!std::is_copy_assignable_v<vk::ExternalImage>);
    static_assert(std::is_move_constructible_v<vk::ExternalImage>);
    static_assert(std::is_move_assignable_v<vk::ExternalImage>);

    const ls::OwnedFd invalid{};
    assert(!invalid);
    assert(invalid.get() == -1);

    int destroyedFd{-1};
    {
        auto [owned, peer] = makePipe();
        destroyedFd = owned.get();
        assert(owned);
        assert(isOpen(destroyedFd));
    }
    assert(!isOpen(destroyedFd));

    {
        auto [source, peer] = makePipe();
        const auto rawFd = source.get();
        ls::OwnedFd destination(std::move(source));
        assert(!source);
        assert(destination.get() == rawFd);
        assert(isOpen(rawFd));
    }

    int replacedFd{-1};
    {
        auto [destination, firstPeer] = makePipe();
        auto [source, secondPeer] = makePipe();
        replacedFd = destination.get();
        const auto transferredFd = source.get();
        destination = std::move(source);
        assert(!source);
        assert(destination.get() == transferredFd);
        assert(!isOpen(replacedFd));
        assert(isOpen(transferredFd));
    }

    int releasedFd{-1};
    {
        auto [owned, peer] = makePipe();
        releasedFd = owned.release();
        assert(!owned);
        assert(isOpen(releasedFd));
    }
    assert(isOpen(releasedFd));
    assert(::close(releasedFd) == 0);

    int resetFd{-1};
    int adoptedFd{-1};
    {
        auto [owned, firstPeer] = makePipe();
        auto [replacement, secondPeer] = makePipe();
        resetFd = owned.get();
        adoptedFd = replacement.release();
        owned.reset(adoptedFd);
        assert(!isOpen(resetFd));
        assert(owned.get() == adoptedFd);
    }
    assert(!isOpen(adoptedFd));

    int successfulImportFd{-1};
    {
        auto [owned, peer] = makePipe();
        successfulImportFd = owned.get();
        int observedFd{-1};
        int importCalls{};
        const auto result = vk::importMemoryWithOwnedFd(owned,
            [&observedFd, &importCalls](int borrowedFd) {
                ++importCalls;
                observedFd = borrowedFd;
                return VK_SUCCESS;
            });
        assert(result == VK_SUCCESS);
        assert(importCalls == 1);
        assert(observedFd == successfulImportFd);
        assert(!owned);
    }
    // The mock did not consume the kernel FD as Vulkan would, so close it explicitly.
    assert(isOpen(successfulImportFd));
    assert(::close(successfulImportFd) == 0);

    int failedImportFd{-1};
    {
        auto [owned, peer] = makePipe();
        failedImportFd = owned.get();
        const auto result = vk::importMemoryWithOwnedFd(owned,
            [](int) { return VK_ERROR_OUT_OF_DEVICE_MEMORY; });
        assert(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        assert(owned);
        assert(isOpen(failedImportFd));
    }
    assert(!isOpen(failedImportFd));

    int throwingImportFd{-1};
    {
        auto [owned, peer] = makePipe();
        throwingImportFd = owned.get();
        try {
            static_cast<void>(vk::importMemoryWithOwnedFd(owned,
                [](int) -> VkResult { throw std::runtime_error("mock import failure"); }));
            assert(false);
        } catch (const std::runtime_error&) {
            assert(owned);
            assert(isOpen(throwingImportFd));
        }
    }
    assert(!isOpen(throwingImportFd));

    {
        auto [owned, peer] = makePipe();
        const auto rawFd = owned.get();
        vk::ExternalImage externalImage{ .fd = std::move(owned) };
        std::vector<vk::ExternalImage> images;
        images.push_back(std::move(externalImage));
        assert(!externalImage.fd);
        assert(images.front().fd.get() == rawFd);
        images.reserve(8);
        assert(images.front().fd.get() == rawFd);
        assert(isOpen(rawFd));
    }

    return 0;
}
