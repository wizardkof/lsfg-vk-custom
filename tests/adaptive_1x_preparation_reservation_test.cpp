#include "adaptive_1x_preparation_reservation.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"

#include <cassert>

using namespace lsfgvk::layer;

int main() {
    uint32_t abortCalls{};
    {
        Adaptive1xPreparationReservation reservation({
            .execute = [](VkSemaphore) -> Adaptive1xPreparedLogicalFinal {
                throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
                    "controlled unavailable executor");
            },
            .abortUnused = [&] { ++abortCalls; },
            .valid = [] { return true; }});
        assert(reservation.valid());
        assert(reservation.state()
            == Adaptive1xPreparationReservation::State::Reserved);
    }
    assert(abortCalls == 1);

    Adaptive1xPreparationReservation source({
        .execute = [](VkSemaphore) -> Adaptive1xPreparedLogicalFinal {
            throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
                "controlled unavailable executor");
        },
        .abortUnused = [&] { ++abortCalls; },
        .valid = [] { return true; }});
    Adaptive1xPreparationReservation moved(std::move(source));
    assert(!source.valid());
    assert(moved.valid());
    try {
        static_cast<void>(moved.execute(reinterpret_cast<VkSemaphore>(0x100)));
        assert(false);
    } catch (const ls::vulkan_error& error) {
        assert(error.error() == VK_ERROR_FEATURE_NOT_PRESENT);
    }
    assert(moved.state()
        == Adaptive1xPreparationReservation::State::CleanlyAborted);

    for (const auto expected : {
            Adaptive1xPreparationReservation::State::Recovered,
            Adaptive1xPreparationReservation::State::ConservativelyRetained}) {
        Adaptive1xPreparationReservation typed({
            .execute = [](VkSemaphore) -> Adaptive1xPreparedLogicalFinal {
                throw ls::vulkan_error(VK_ERROR_DEVICE_LOST,
                    "controlled terminal outcome");
            },
            .abortUnused = [] {},
            .valid = [] { return true; },
            .failureState = [expected] { return expected; }});
        try {
            static_cast<void>(typed.execute(
                reinterpret_cast<VkSemaphore>(uintptr_t{0x101})));
            assert(false);
        } catch (const ls::vulkan_error&) {}
        assert(typed.state() == expected);
    }
    assert(abortCalls == 1);

    uint32_t attachCalls{};
    auto makeAttachable = [&] {
        return Adaptive1xPreparationReservation({
            .execute = [](VkSemaphore) -> Adaptive1xPreparedLogicalFinal {
                throw ls::vulkan_error(VK_ERROR_FEATURE_NOT_PRESENT,
                    "controlled unavailable executor");
            },
            .abortUnused = [] {},
            .valid = [] { return true; },
            .attachWaitBacking = [&](std::shared_ptr<void>) {
                ++attachCalls; return true;
            }});
    };
    auto attachedSource = makeAttachable();
    assert(attachedSource.attachWaitBacking(std::make_shared<int>(1)));
    Adaptive1xPreparationReservation attachedMoved(std::move(attachedSource));
    assert(!attachedSource.valid() && attachedMoved.valid());
    assert(!attachedMoved.attachWaitBacking(std::make_shared<int>(2)));
    auto assigned = makeAttachable();
    auto attachedForAssignment = makeAttachable();
    assert(attachedForAssignment.attachWaitBacking(std::make_shared<int>(3)));
    assigned = std::move(attachedForAssignment);
    assert(!attachedForAssignment.valid() && assigned.valid());
    assert(!assigned.attachWaitBacking(std::make_shared<int>(4)));
    assert(attachCalls == 2);
    return 0;
}
