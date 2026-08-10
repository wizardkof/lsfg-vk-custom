/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "virtual_swapchain_state.hpp"

#include <algorithm>
#include <chrono>

using namespace lsfgvk::layer;

VirtualSwapchainState::VirtualSwapchainState(size_t imageCount) :
        images(imageCount, ImageState::Available) {}

size_t VirtualSwapchainState::imageCount() const noexcept {
    return this->images.size();
}

std::optional<uint32_t> VirtualSwapchainState::acquireUnlocked() {
    for (size_t i = 0; i < this->images.size(); ++i) {
        if (this->images.at(i) != ImageState::Available)
            continue;

        this->images.at(i) = ImageState::Acquired;
        return static_cast<uint32_t>(i);
    }

    return std::nullopt;
}

std::optional<uint32_t> VirtualSwapchainState::tryAcquire() {
    std::scoped_lock lock(this->mutex);
    if (this->stopping)
        return std::nullopt;
    return this->acquireUnlocked();
}

std::optional<uint32_t> VirtualSwapchainState::waitAcquire(Duration timeout) {
    std::unique_lock lock(this->mutex);
    const auto ready = [this]() {
        return this->stopping
            || std::ranges::find(this->images, ImageState::Available) != this->images.end();
    };

    if (timeout == Duration::max()) {
        this->cv.wait(lock, ready);
    } else if (!this->cv.wait_for(lock, timeout, ready)) {
        return std::nullopt;
    }

    if (this->stopping)
        return std::nullopt;
    return this->acquireUnlocked();
}

bool VirtualSwapchainState::release(uint32_t imageIndex) {
    std::scoped_lock lock(this->mutex);
    if (imageIndex >= this->images.size()
            || this->images.at(imageIndex) != ImageState::Acquired)
        return false;

    this->images.at(imageIndex) = ImageState::Available;
    this->cv.notify_all();
    return true;
}

bool VirtualSwapchainState::queuePresent(uint32_t imageIndex, uint64_t serial) {
    std::scoped_lock lock(this->mutex);
    if (this->stopping
            || imageIndex >= this->images.size()
            || this->images.at(imageIndex) != ImageState::Acquired)
        return false;

    this->images.at(imageIndex) = ImageState::Queued;
    this->presents.push_back(Present {
        .imageIndex = imageIndex,
        .serial = serial
    });
    this->cv.notify_all();
    return true;
}

std::optional<VirtualSwapchainState::Present>
VirtualSwapchainState::waitPresent(Duration timeout) {
    std::unique_lock lock(this->mutex);
    const auto ready = [this]() {
        return this->stopping || !this->presents.empty();
    };

    if (timeout == Duration::max()) {
        this->cv.wait(lock, ready);
    } else if (!this->cv.wait_for(lock, timeout, ready)) {
        return std::nullopt;
    }

    if (this->stopping || this->presents.empty())
        return std::nullopt;

    const auto present = this->presents.front();
    this->presents.pop_front();

    if (present.imageIndex >= this->images.size()
            || this->images.at(present.imageIndex) != ImageState::Queued)
        return std::nullopt;

    this->images.at(present.imageIndex) = ImageState::Presenting;
    return present;
}

bool VirtualSwapchainState::complete(uint32_t imageIndex) {
    std::scoped_lock lock(this->mutex);
    if (imageIndex >= this->images.size()
            || this->images.at(imageIndex) != ImageState::Presenting)
        return false;

    this->images.at(imageIndex) = ImageState::Available;
    this->cv.notify_all();
    return true;
}

void VirtualSwapchainState::stop() {
    std::scoped_lock lock(this->mutex);
    this->stopping = true;
    this->cv.notify_all();
}

bool VirtualSwapchainState::stopped() const {
    const std::scoped_lock lock(this->mutex);
    return this->stopping;
}
