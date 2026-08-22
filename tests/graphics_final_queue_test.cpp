#include "graphics_final_queue.hpp"

#include <cassert>
#include <cstdint>

using namespace lsfgvk::layer;

namespace {
template<class T> T h(uintptr_t value) { return reinterpret_cast<T>(value); }
}

int main() {
    const auto queue = h<VkQueue>(0x1234);
    GraphicsFinalQueueInfo compute{
        queue, 7, 0, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, true, true};
    assert(graphicsFinalExecutionMode(compute, true)
        == GraphicsFinalExecutionMode::INELIGIBLE);

    GraphicsFinalQueueInfo borrowed{
        queue, 3, 0, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT, true, true};
    assert(graphicsFinalExecutionMode(borrowed, true)
        == GraphicsFinalExecutionMode::BORROWED_SYNCHRONOUS);
    assert(graphicsFinalExecutionMode(borrowed, false)
        == GraphicsFinalExecutionMode::INELIGIBLE);
    borrowed.surfacePresentSupported = false;
    assert(graphicsFinalExecutionMode(borrowed, true)
        == GraphicsFinalExecutionMode::INELIGIBLE);

    GraphicsFinalQueueInfo dedicated{
        queue, 3, 1, VK_QUEUE_GRAPHICS_BIT, true, false};
    assert(graphicsFinalExecutionMode(dedicated, false)
        == GraphicsFinalExecutionMode::DEDICATED_ASYNC);

    BorrowedGraphicsQueueLease lease(queue, 3, 0, 41);
    assert(lease.validFor(queue, 3));
    assert(!lease.validFor(queue, 7));
    assert(lease.operation() == 41 && lease.index() == 0);
    assert(lease.release());
    assert(!lease.validFor(queue, 3));
    assert(!lease.release());
    assert(lease.currentState() == BorrowedGraphicsQueueLease::State::RELEASED);
    return 0;
}
