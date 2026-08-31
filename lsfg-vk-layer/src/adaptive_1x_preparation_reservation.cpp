/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "adaptive_1x_preparation_reservation.hpp"

#include "lsfg-vk-common/helpers/errors.hpp"

namespace lsfgvk::layer {

Adaptive1xPreparationReservation::Adaptive1xPreparationReservation(
        Operations value) noexcept : ops(std::move(value)) {}

Adaptive1xPreparationReservation::Adaptive1xPreparationReservation(
        Adaptive1xPreparationReservation&& other) noexcept
    : ops(std::move(other.ops)), stateValue(other.stateValue),
      waitBackingAttached(other.waitBackingAttached) {
    other.stateValue = State::CleanlyAborted;
    other.waitBackingAttached = false;
}

Adaptive1xPreparationReservation& Adaptive1xPreparationReservation::operator=(
        Adaptive1xPreparationReservation&& other) noexcept {
    if (this == &other) return *this;
    abortUnused();
    ops = std::move(other.ops);
    stateValue = other.stateValue;
    waitBackingAttached = other.waitBackingAttached;
    other.stateValue = State::CleanlyAborted;
    other.waitBackingAttached = false;
    return *this;
}

Adaptive1xPreparationReservation::~Adaptive1xPreparationReservation() noexcept {
    abortUnused();
}

bool Adaptive1xPreparationReservation::valid() const noexcept {
    return stateValue == State::Reserved && ops.execute && ops.abortUnused
        && ops.valid && ops.valid();
}

void Adaptive1xPreparationReservation::abortUnused() noexcept {
    if (stateValue != State::Reserved) return;
    if (ops.abortUnused) ops.abortUnused();
    stateValue = State::CleanlyAborted;
}

Adaptive1xPreparedLogicalFinal Adaptive1xPreparationReservation::execute(
        VkSemaphore wait) {
    if (!valid() || wait == VK_NULL_HANDLE)
        throw ls::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
            "Adaptive 1x preparation reservation is not executable");
    stateValue = State::ExecuteCalled;
    try {
        auto prepared = ops.execute(wait);
        if (!prepared.valid())
            throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                "Adaptive 1x preparation transfer is invalid");
        stateValue = State::PreparedTransferred;
        return prepared;
    } catch (...) {
        stateValue = ops.failureState ? ops.failureState()
                                      : State::CleanlyAborted;
        throw;
    }
}

bool Adaptive1xPreparationReservation::attachWaitBacking(
        std::shared_ptr<void> backing) noexcept {
    if (stateValue != State::Reserved || waitBackingAttached || !backing
            || !ops.attachWaitBacking || !ops.attachWaitBacking(std::move(backing)))
        return false;
    waitBackingAttached = true;
    return true;
}

} // namespace lsfgvk::layer
