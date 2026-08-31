/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "adaptive_1x_prepared_logical_final.hpp"

#include <cstdint>
#include <functional>

namespace lsfgvk::layer {

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
#define LSFGVK_ADAPTIVE_RESERVATION_VISIBILITY __attribute__((visibility("default")))
#else
#define LSFGVK_ADAPTIVE_RESERVATION_VISIBILITY
#endif
class LSFGVK_ADAPTIVE_RESERVATION_VISIBILITY Adaptive1xPreparationReservation final {
public:
    enum class State : uint8_t {
        Reserved, ExecuteCalled, AcquireAccepted, PhysicalIdentityBound,
        ProducerPrepared, ProducerAccepted, PreparedTransferred,
        CleanlyAborted, Recovered, ConservativelyRetained
    };
    struct Operations final {
        std::function<Adaptive1xPreparedLogicalFinal(VkSemaphore)> execute;
        std::function<void()> abortUnused;
        std::function<bool()> valid;
        std::function<State()> failureState;
        std::function<bool(std::shared_ptr<void>)> attachWaitBacking;
    };

    explicit Adaptive1xPreparationReservation(Operations) noexcept;
    Adaptive1xPreparationReservation(
        const Adaptive1xPreparationReservation&) = delete;
    Adaptive1xPreparationReservation& operator=(
        const Adaptive1xPreparationReservation&) = delete;
    Adaptive1xPreparationReservation(
        Adaptive1xPreparationReservation&&) noexcept;
    Adaptive1xPreparationReservation& operator=(
        Adaptive1xPreparationReservation&&) noexcept;
    ~Adaptive1xPreparationReservation() noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] State state() const noexcept { return stateValue; }
    [[nodiscard]] bool attachWaitBacking(std::shared_ptr<void>) noexcept;
    [[nodiscard]] Adaptive1xPreparedLogicalFinal execute(VkSemaphore);

private:
    void abortUnused() noexcept;
    Operations ops;
    State stateValue{State::Reserved};
    bool waitBackingAttached{};
};
#undef LSFGVK_ADAPTIVE_RESERVATION_VISIBILITY

} // namespace lsfgvk::layer
