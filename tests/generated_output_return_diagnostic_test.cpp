#include "generated_output_return_diagnostic.hpp"
#include "swapchain.hpp"
#include "lsfg-vk-common/fnv1a.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace lsfgvk::backend {
struct RuntimeGeneratedFrameTokenTestAccess {
    static RuntimeGeneratedFrameToken issue(RuntimeGenerationId generation, VkImage image,
            const std::shared_ptr<const uint8_t>& lifetime) {
        return RuntimeGeneratedFrameToken(generation, image, {8, 8},
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 3, lifetime);
    }
};
struct RuntimeGenerateDiagnosticPendingTestAccess {
    static RuntimeGenerateDiagnosticPending issue(RuntimeGenerationId generation, VkImage image,
            VkSemaphore readiness, const std::shared_ptr<const uint8_t>& lifetime) {
        return RuntimeGenerateDiagnosticPending(generation, image, {8, 8},
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 3,
            readiness, lifetime);
    }
};
}

namespace {
vk::PhysicalDeviceIdentity identity(uint8_t id) {
    vk::PhysicalDeviceIdentity result;
    result.name = id == 1 ? "render" : "generation";
    result.deviceUuid.back() = id;
    result.driverUuid.back() = static_cast<uint8_t>(id + 10);
    return result;
}
}
#include <type_traits>

using namespace lsfgvk::layer;

int main() {
    static_assert(!std::is_copy_constructible_v<RuntimeGeneratedBReturnPending>);
    static_assert(std::is_move_constructible_v<RuntimeGeneratedBReturnPending>);
    assert(returnSubmissionFence(ReturnSubmissionFencePolicy::ACTIVE_COMPATIBLE,
        reinterpret_cast<VkFence>(uintptr_t{1})) == VK_NULL_HANDLE);
    assert(returnSubmissionFence(ReturnSubmissionFencePolicy::SHADOW_REAL,
        reinterpret_cast<VkFence>(uintptr_t{1})) != VK_NULL_HANDLE);
    static_assert(selectGeneratedOutputTerminalConsumer(false, false)
        == GeneratedOutputTerminalConsumer::DiagnosticOnly);
    static_assert(selectGeneratedOutputTerminalConsumer(true, false)
        == GeneratedOutputTerminalConsumer::D3B1);
    static_assert(selectGeneratedOutputTerminalConsumer(false, true)
        == GeneratedOutputTerminalConsumer::D3B2);
    static_assert(selectGeneratedOutputTerminalConsumer(true, true)
        == GeneratedOutputTerminalConsumer::D3B2);
    static_assert(selectGeneratedOutputTerminalConsumer(true, true, true)
        == GeneratedOutputTerminalConsumer::D3B3);
    static_assert(captureDiagnosticFormatSupported(VK_FORMAT_B8G8R8A8_UNORM));
    static_assert(captureDiagnosticFormatSupported(VK_FORMAT_R8G8B8A8_UNORM));
    static_assert(captureDiagnosticFormatSupported(VK_FORMAT_A2R10G10B10_UNORM_PACK32));
    static_assert(!captureDiagnosticFormatSupported(VK_FORMAT_R16G16B16A16_SFLOAT));
    ::unsetenv("LSFGVK_D3B1_PRESENT_DIAGNOSTIC");
    assert(!d3b1PresentationDiagnosticEnabled());
    ::setenv("LSFGVK_D3B1_PRESENT_DIAGNOSTIC", "0", 1);
    assert(!d3b1PresentationDiagnosticEnabled());
    ::setenv("LSFGVK_D3B1_PRESENT_DIAGNOSTIC", "1", 1);
    assert(d3b1PresentationDiagnosticEnabled());
    ::setenv("LSFGVK_D3B1_PRESENT_DIAGNOSTIC", "true", 1);
    assert(!d3b1PresentationDiagnosticEnabled());
    ::unsetenv("LSFGVK_D3B1_PRESENT_DIAGNOSTIC");
    static_assert(!std::is_copy_constructible_v<vk::RuntimeForeignImageReadbackPending>);
    static_assert(!std::is_copy_assignable_v<vk::RuntimeForeignImageReadbackPending>);
    static_assert(std::is_move_constructible_v<vk::RuntimeForeignImageReadbackPending>);
    static_assert(std::is_move_assignable_v<vk::RuntimeForeignImageReadbackPending>);
    vk::RuntimeForeignImageReadbackPending emptyReadback;
    assert(!emptyReadback.valid());
    auto movedEmptyReadback = std::move(emptyReadback);
    assert(!emptyReadback.valid() && !movedEmptyReadback.valid());
    const auto noHandoff = vk::RuntimeForeignImageHandoffInfo{};
    assert(noHandoff.destinationQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    assert(noHandoff.signalSemaphore == VK_NULL_HANDLE);
    const auto differentFamily = vk::RuntimeForeignImageHandoffInfo{
        .destinationQueueFamilyIndex = 7,
        .signalSemaphore = reinterpret_cast<VkSemaphore>(uintptr_t{0x7890})};
    assert(differentFamily.destinationQueueFamilyIndex != 3);
    assert(differentFamily.signalSemaphore != VK_NULL_HANDLE);
    static_assert(D3B1PresentationState::PASS != D3B1PresentationState::FAILED);
    std::vector<uint8_t> known(256U * 256U * 4U);
    for (size_t i = 0; i < known.size(); i += 4) {
        known[i] = 0xff; known[i + 1] = 0; known[i + 2] = 0xff; known[i + 3] = 0xff;
    }
    assert(lsfgvk::common::fnv1a64(known.data(), known.size()) == 0x4674de733bca2325ULL);
    assert(!generatedOutputNeedsDeviceIdle(GeneratedOutputCleanupState::NO_SUBMIT));
    assert(generatedOutputNeedsDeviceIdle(GeneratedOutputCleanupState::B_SUBMITTED));
    assert(generatedOutputNeedsDeviceIdle(GeneratedOutputCleanupState::A_SUBMITTED));
    assert(!generatedOutputNeedsDeviceIdle(GeneratedOutputCleanupState::A_COMPLETED));

    auto render = identity(1); auto generation = identity(2);
    auto pair = vk::bindRuntimeDevicePair(render, generation, "generation");
    assert(pair.has_value());
    assert(generatedOutputRoleBindingEligible(*pair, generation, render, true));
    assert(!generatedOutputRoleBindingEligible(*pair, generation, render, false));
    assert(!generatedOutputRoleBindingEligible(*pair, render, generation, true));
    auto same = vk::bindRuntimeDevicePair(render, render, std::nullopt);
    assert(same.has_value());
    assert(!generatedOutputRoleBindingEligible(*same, render, render, true));
    static_assert(!std::is_copy_constructible_v<lsfgvk::backend::RuntimeGeneratedFrameToken>);
    static_assert(std::is_move_constructible_v<lsfgvk::backend::RuntimeGeneratedFrameToken>);
    static_assert(!std::is_copy_constructible_v<lsfgvk::backend::RuntimeGenerateDiagnosticPending>);
    static_assert(std::is_move_constructible_v<lsfgvk::backend::RuntimeGenerateDiagnosticPending>);
    static_assert(std::is_default_constructible_v<lsfgvk::backend::RuntimeGenerateDiagnosticPending>);
    lsfgvk::backend::RuntimeGenerateDiagnosticPending emptyPending;
    assert(!emptyPending.valid());
    auto movedEmptyPending = std::move(emptyPending);
    assert(!emptyPending.valid() && !movedEmptyPending.valid());

    const auto reusedImage = reinterpret_cast<VkImage>(uintptr_t{0x1234});
    auto pendingLifetime1 = std::make_shared<const uint8_t>(0);
    const auto fakeReady = reinterpret_cast<VkSemaphore>(uintptr_t{0x5678});
    auto pendingCapability =
        lsfgvk::backend::RuntimeGenerateDiagnosticPendingTestAccess::issue(
            44, reusedImage, fakeReady, pendingLifetime1);
    assert(pendingCapability.valid());
    assert(pendingCapability.identity() == 44);
    assert(pendingCapability.readinessSemaphore() == fakeReady);
    auto movedPending = std::move(pendingCapability);
    assert(!pendingCapability.valid() && movedPending.valid());
    auto retirementAuthority = movedPending.operationRetirementAuthority();
    assert(retirementAuthority.valid() && retirementAuthority.generationId() == 44);
    bool duplicateAuthorityRejected = false;
    try { static_cast<void>(movedPending.operationRetirementAuthority()); }
    catch (const std::logic_error&) { duplicateAuthorityRejected = true; }
    assert(duplicateAuthorityRejected);
    movedPending.consumeTransport();
    bool pendingRetryRejected = false;
    try { movedPending.consumeTransport(); }
    catch (const std::logic_error&) { pendingRetryRejected = true; }
    assert(pendingRetryRejected);
    pendingLifetime1.reset();
    assert(!movedPending.valid());
    auto pendingLifetime2 = std::make_shared<const uint8_t>(0);
    auto recreatedPending =
        lsfgvk::backend::RuntimeGenerateDiagnosticPendingTestAccess::issue(
            45, reusedImage, fakeReady, pendingLifetime2);
    assert(!movedPending.valid() && recreatedPending.valid());

    auto lifetime1 = std::make_shared<const uint8_t>(0);
    auto stale = lsfgvk::backend::RuntimeGeneratedFrameTokenTestAccess::issue(
        41, reusedImage, lifetime1);
    assert(stale.valid());
    lifetime1.reset();
    assert(!stale.valid());
    bool staleConsumeRejected = false;
    try { stale.consume(); }
    catch (const std::logic_error&) { staleConsumeRejected = true; }
    assert(staleConsumeRejected);

    auto lifetime2 = std::make_shared<const uint8_t>(0);
    auto recreated = lsfgvk::backend::RuntimeGeneratedFrameTokenTestAccess::issue(
        42, reusedImage, lifetime2);
    assert(!stale.valid() && recreated.valid());

    auto moved = std::move(recreated);
    assert(!recreated.valid() && moved.valid());
    moved.consume();
    assert(!moved.valid());
    bool secondConsumeRejected = false;
    try { moved.consume(); }
    catch (const std::logic_error&) { secondConsumeRejected = true; }
    assert(secondConsumeRejected);

    auto failedAttempt = lsfgvk::backend::RuntimeGeneratedFrameTokenTestAccess::issue(
        43, reusedImage, lifetime2);
    GeneratedOutputReturnStateMachine failedReturn;
    failedReturn.advance(GeneratedOutputReturnState::EMPTY,
        GeneratedOutputReturnState::D2_VALIDATED);
    failedAttempt.consume();
    failedReturn.advance(GeneratedOutputReturnState::D2_VALIDATED,
        GeneratedOutputReturnState::TOKEN_CONSUMED);
    failedReturn.fail();
    assert(!failedAttempt.valid() && !failedReturn.markerReady());
    bool retryRejected = false;
    try { failedAttempt.consume(); }
    catch (const std::logic_error&) { retryRejected = true; }
    assert(retryRejected);

    GeneratedOutputReturnStateMachine states;
    assert(!states.markerReady());
    const GeneratedOutputReturnState order[]{GeneratedOutputReturnState::D2_VALIDATED,
        GeneratedOutputReturnState::TOKEN_CONSUMED,
        GeneratedOutputReturnState::B_BACKING_READY,
        GeneratedOutputReturnState::B_COPY_SUBMITTED,
        GeneratedOutputReturnState::B_SYNC_EXPORTED,
        GeneratedOutputReturnState::A_IMPORTED,
        GeneratedOutputReturnState::A_COPY_SUBMITTED,
        GeneratedOutputReturnState::A_VALIDATED,
        GeneratedOutputReturnState::PASS};
    auto previous = GeneratedOutputReturnState::EMPTY;
    for (const auto next : order) { states.advance(previous, next); previous = next; }
    assert(states.markerReady());
    bool secondPassRejected = false;
    try { states.advance(GeneratedOutputReturnState::PASS, GeneratedOutputReturnState::PASS); }
    catch (const std::logic_error&) { secondPassRejected = true; }
    assert(secondPassRejected);

    const GeneratedOutputReturnState allPrePass[] = {
        GeneratedOutputReturnState::EMPTY, GeneratedOutputReturnState::D2_VALIDATED,
        GeneratedOutputReturnState::TOKEN_CONSUMED, GeneratedOutputReturnState::B_BACKING_READY,
        GeneratedOutputReturnState::B_COPY_SUBMITTED, GeneratedOutputReturnState::B_SYNC_EXPORTED,
        GeneratedOutputReturnState::A_IMPORTED, GeneratedOutputReturnState::A_COPY_SUBMITTED};
    for (const auto state : allPrePass) {
        GeneratedOutputReturnStateMachine probe;
        if (state != GeneratedOutputReturnState::EMPTY) {
            auto prior = GeneratedOutputReturnState::EMPTY;
            for (int next = 1; next <= static_cast<int>(state); ++next) {
                auto current = static_cast<GeneratedOutputReturnState>(next);
                probe.advance(prior, current); prior = current;
            }
        }
        assert(!probe.markerReady());
    }

    GeneratedOutputReturnStateMachine skipped;
    bool skipRejected = false;
    try { skipped.advance(GeneratedOutputReturnState::EMPTY,
        GeneratedOutputReturnState::B_BACKING_READY); }
    catch (const std::logic_error&) { skipRejected = true; }
    assert(skipRejected);
    skipped.fail();
    bool failedTerminal = false;
    try { skipped.advance(GeneratedOutputReturnState::FAILED,
        GeneratedOutputReturnState::D2_VALIDATED); }
    catch (const std::logic_error&) { failedTerminal = true; }
    assert(failedTerminal && !skipped.markerReady());

    const GeneratedOutputIntegrity expected{7, {8, 4}, VK_FORMAT_R8G8B8A8_UNORM,
        128, 95, 0x1234};
    assert(generatedOutputIntegrityMatches(expected, expected));
    auto mismatch = expected; mismatch.generation = 8;
    assert(!generatedOutputIntegrityMatches(expected, mismatch));
    mismatch = expected; mismatch.byteCount = 127;
    assert(!generatedOutputIntegrityMatches(expected, mismatch));
    mismatch = expected; mismatch.nonzeroByteCount = 94;
    assert(!generatedOutputIntegrityMatches(expected, mismatch));
    mismatch = expected; mismatch.checksum = 0x1235;
    assert(!generatedOutputIntegrityMatches(expected, mismatch));
    mismatch = expected; mismatch.format = VK_FORMAT_B8G8R8A8_UNORM;
    assert(!generatedOutputIntegrityMatches(expected, mismatch));
    mismatch = expected; mismatch.extent.width = 9;
    assert(!generatedOutputIntegrityMatches(expected, mismatch));

    GpuChainedReturnStateMachine chained;
    const GpuChainedReturnState chainedOrder[]{
        GpuChainedReturnState::D2_SUBMITTED,
        GpuChainedReturnState::PENDING_GENERATION,
        GpuChainedReturnState::TRANSPORT_CONSUMED,
        GpuChainedReturnState::RETURN_B_PREPARED,
        GpuChainedReturnState::RETURN_B_SUBMITTED,
        GpuChainedReturnState::A_SUBMITTED,
        GpuChainedReturnState::A_COMPLETED,
        GpuChainedReturnState::B_VALIDATED,
        GpuChainedReturnState::A_VALIDATED,
        GpuChainedReturnState::PASS};
    auto chainedPrevious = GpuChainedReturnState::EMPTY;
    for (const auto next : chainedOrder) {
        chained.advance(chainedPrevious, next); chainedPrevious = next;
    }
    assert(chained.markerReady());
    GpuChainedReturnStateMachine chainedFailure;
    chainedFailure.advance(GpuChainedReturnState::EMPTY,
        GpuChainedReturnState::D2_SUBMITTED);
    chainedFailure.fail();
    assert(!chainedFailure.markerReady());

    const GpuChainedMarkerGate exact{true, false, false, true, true, true, false};
    assert(exact.ready());
    auto gate = exact; gate.readinessDependencyUsed = false; assert(!gate.ready());
    gate = exact; gate.generateReturnHostWaitUsed = true; assert(!gate.ready());
    gate = exact; gate.bToAHostWaitUsed = true; assert(!gate.ready());
    gate = exact; gate.bValidated = false; assert(!gate.ready());
    gate = exact; gate.aValidated = false; assert(!gate.ready());
    gate = exact; gate.integrityExact = false; assert(!gate.ready());
    gate = exact; gate.synchronousD3A1Path = true; assert(!gate.ready());
}
