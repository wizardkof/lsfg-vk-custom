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
    ++this->wakeCounter;
    this->cv.notify_all();
    return true;
}

bool VirtualSwapchainState::queuePresent(uint32_t imageIndex, uint64_t serial) {
    auto prepared = this->preparePresent(imageIndex, serial);
    return prepared && this->commitPreparedPresent(*prepared);
}

std::optional<VirtualSwapchainState::PreparedPresent>
VirtualSwapchainState::preparePresent(uint32_t imageIndex, uint64_t serial) {
    std::scoped_lock lock(this->mutex);
    if (this->stopping
            || imageIndex >= this->images.size()
            || this->images.at(imageIndex) != ImageState::Acquired)
        return std::nullopt;

    PreparedPresent prepared;
    prepared.node.push_back(Present {
        .imageIndex = imageIndex,
        .serial = serial
    });
    return prepared;
}

bool VirtualSwapchainState::commitPreparedPresent(
        PreparedPresent& prepared) noexcept {
    std::scoped_lock lock(this->mutex);
    if (prepared.node.size() != 1)
        return false;
    const auto& present = prepared.node.front();
    if (this->stopping
            || present.imageIndex >= this->images.size()
            || this->images.at(present.imageIndex) != ImageState::Acquired)
        return false;

    this->images.at(present.imageIndex) = ImageState::Queued;
    this->presents.splice(this->presents.end(), prepared.node);
    ++this->wakeCounter;
    this->cv.notify_all();
    return true;
}

std::optional<VirtualSwapchainState::PreparedBatchPresent>
VirtualSwapchainState::prepareBatchPresent(
        uint32_t imageIndex, uint64_t serial) noexcept {
    const std::scoped_lock lock(this->mutex);
    if (this->stopping || serial == 0 || imageIndex >= this->images.size()
            || this->images.at(imageIndex) != ImageState::Acquired)
        return std::nullopt;
    this->images.at(imageIndex) = ImageState::BatchPrepared;
    return PreparedBatchPresent{imageIndex, serial, true};
}

bool VirtualSwapchainState::commitBatchPresent(
        PreparedBatchPresent& prepared) noexcept {
    const std::scoped_lock lock(this->mutex);
    if (!prepared.valid || this->stopping
            || prepared.imageIndex >= this->images.size()
            || this->images.at(prepared.imageIndex) != ImageState::BatchPrepared)
        return false;
    this->images.at(prepared.imageIndex) = ImageState::Presenting;
    return true;
}

bool VirtualSwapchainState::abortBatchPresent(
        PreparedBatchPresent& prepared, bool bridgeAccepted) noexcept {
    const std::scoped_lock lock(this->mutex);
    if (!prepared.valid || prepared.imageIndex >= this->images.size())
        return false;
    const auto expected = bridgeAccepted ? ImageState::Presenting
                                         : ImageState::BatchPrepared;
    if (this->images.at(prepared.imageIndex) != expected) return false;
    this->images.at(prepared.imageIndex) = bridgeAccepted
        ? ImageState::Available : ImageState::Acquired;
    prepared.valid = false;
    ++this->wakeCounter;
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
    ++this->wakeCounter;
    this->cv.notify_all();
    return true;
}

uint64_t VirtualSwapchainState::wakeGeneration() const noexcept {
    const std::scoped_lock lock(this->mutex);
    return this->wakeCounter;
}

void VirtualSwapchainState::waitForWake(
        uint64_t observedGeneration, Duration timeout) {
    std::unique_lock lock(this->mutex);
    const auto ready = [this, observedGeneration]() {
        return this->stopping || !this->presents.empty()
            || this->wakeCounter != observedGeneration;
    };
    if (timeout == Duration::max())
        this->cv.wait(lock, ready);
    else
        static_cast<void>(this->cv.wait_for(lock, timeout, ready));
}

void VirtualSwapchainState::waitForWake(Duration timeout) {
    waitForWake(wakeGeneration(), timeout);
}

void VirtualSwapchainState::wake() {
    const std::scoped_lock lock(this->mutex);
    ++this->wakeCounter;
    this->cv.notify_all();
}

void VirtualSwapchainState::stop() {
    std::scoped_lock lock(this->mutex);
    this->stopping = true;
    ++this->wakeCounter;
    this->cv.notify_all();
}

bool VirtualSwapchainState::stopped() const {
    const std::scoped_lock lock(this->mutex);
    return this->stopping;
}
