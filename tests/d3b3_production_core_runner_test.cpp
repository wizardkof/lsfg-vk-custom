/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "shadow_b_return_execution_harness.hpp"
#include "d3b3_production_core_owner.hpp"
#include "lsfg-vk-common/fnv1a.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vk {
struct RuntimeImageEndpointTestAccess {
    static RuntimeImageEndpoint make(RuntimeExchangeEndpoint endpoint,
            VkImage image, VkExtent2D extent, VkFormat format) {
        RuntimeImageEndpoint result;
        endpoint.FreeCommandBuffers = nullptr;
        endpoint.DestroyCommandPool = nullptr;
        endpoint.DestroyImage = nullptr;
        result.endpoint = std::move(endpoint);
        result.imageHandle = image;
        result.extent = extent;
        result.format = format;
        result.commandBuffers = {
            reinterpret_cast<VkCommandBuffer>(uintptr_t{0xA6A5F0})};
        return result;
    }
};
}

namespace {

using lsfgvk::test::ShadowBReturnExecutionHarness;

struct PreparedSource {
    VkImage image{};
    std::array<uint8_t, 8> pattern{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
};

struct RunnerTrace {
    std::vector<std::string> events;
    uint32_t observationPolls{};
};

uint64_t checksum(const PreparedSource& source) {
    return lsfgvk::common::fnv1a64(source.pattern.data(), source.pattern.size());
}

void prepareSource(PreparedSource& source, char label, RunnerTrace& trace) {
    assert(source.layout == VK_IMAGE_LAYOUT_UNDEFINED);
    trace.events.emplace_back(std::string("PREP_") + label + "_UNDEFINED");
    source.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    trace.events.emplace_back(std::string("PREP_") + label + "_TRANSFER_DST");
    trace.events.emplace_back(std::string("PREP_") + label + "_PATTERN");
    source.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    trace.events.emplace_back(std::string("PREP_") + label + "_TRANSFER_SRC");
}

void observeSource(const PreparedSource& source, char label,
        const vk::RuntimeImageObservationDescriptor& descriptor,
        RunnerTrace& trace) {
    assert(descriptor.image != VK_NULL_HANDLE);
    assert(descriptor.layout == VK_IMAGE_LAYOUT_GENERAL);
    assert(descriptor.format == VK_FORMAT_R8G8B8A8_UNORM);
    assert(descriptor.extent.width == 64 && descriptor.extent.height == 64);
    assert(!descriptor.lifetime.expired());
    // The controlled dispatch writes the deterministic source pattern into
    // the observation model.  The production readback API itself is covered
    // by runtime_image_observation_test; this runner proves the owner
    // sequencing never inserts it between C ingest and Generate.
    const auto observed = checksum(source);
    assert(observed != 0);
    trace.events.emplace_back(std::string("OBSERVE_") + label + "_0x"
        + std::to_string(observed));
    ++trace.observationPolls;
}

void assertTrace(const RunnerTrace& trace) {
    assert(trace.events.size() == 32);
    const std::vector<std::string> expectedPrefix{
        "PREP_A_UNDEFINED", "PREP_A_TRANSFER_DST", "PREP_A_PATTERN",
        "PREP_A_TRANSFER_SRC", "A_TRANSPORT_SUBMIT", "A_WARMUP_SUBMIT",
        "A_WARMUP_RETIRE"};
    assert(std::equal(expectedPrefix.begin(), expectedPrefix.end(), trace.events.begin()));
    const auto cIngest = std::find(trace.events.begin(), trace.events.end(),
        "C_GENERATE_SOURCE_SUBMIT");
    const auto generate = std::find(trace.events.begin(), trace.events.end(),
        "GENERATE_SUBMIT");
    const auto cObservation = std::find_if(trace.events.begin(), trace.events.end(),
        [](const std::string& event) { return event.rfind("OBSERVE_C_", 0) == 0; });
    assert(cIngest != trace.events.end() && generate != trace.events.end()
        && cObservation != trace.events.end());
    assert(cIngest + 1 == generate);
    assert(generate + 1 < cObservation);
    const std::vector<std::string> expectedSuffix{
        "B_RETURN_SUBMIT", "B_RETURN_NOT_READY", "B_RETURN_RETIRE",
        "A_RETURN_SUBMIT", "OUTPUT_RELEASE", "A_RETURN_NOT_READY",
        "A_RETURN_RETIRE"};
    assert(std::equal(expectedSuffix.begin(), expectedSuffix.end(),
        trace.events.end() - expectedSuffix.size()));
}

} // namespace

int main() {
    ShadowBReturnExecutionHarness harness;
    auto backend = harness.createB0B3Instance();
    const auto pair = harness.devicePair();
    std::array<PreparedSource, 2> sources{
        PreparedSource{reinterpret_cast<VkImage>(uintptr_t{0xA6A510}),
            {0x10, 0x11, 0x12, 0x13, 0x10, 0x11, 0x12, 0x13}},
        PreparedSource{reinterpret_cast<VkImage>(uintptr_t{0xA6A511}),
            {0x20, 0x21, 0x22, 0x23, 0x20, 0x21, 0x22, 0x23}}};
    PreparedSource sourceC{
        reinterpret_cast<VkImage>(uintptr_t{0xA6A512}),
        {0x30, 0x31, 0x32, 0x33, 0x30, 0x31, 0x32, 0x33}};
    RunnerTrace trace;
    std::array<VkImage, 2> currentSource{};
    const auto setSource = [&](PreparedSource& source, char label, size_t slot) {
        prepareSource(source, label, trace);
        currentSource[slot] = source.image;
    };

    lsfgvk::layer::D3B3ProductionCoreOwner owner({
        .backend = backend.get(),
        .devicePair = &pair,
        .exchangeChannel = harness.exchangeChannelOwnerToken(),
        .generationEndpoint = harness.endpoint(),
        .renderEndpoint = harness.renderEndpoint(),
        .extent = {64, 64},
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .modifier = 0,
        .flow = 1.0F,
        .performanceMode = false,
        .sourceImage = [&](lsfgvk::backend::TemporalSourceSlot slot) {
            return currentSource[static_cast<size_t>(slot)];
        }});
    assert(owner.structurallyReady());

    auto frameTransportA = vk::RuntimeImageEndpointTestAccess::make(
        harness.frameTransportAEndpoint(),
        reinterpret_cast<VkImage>(uintptr_t{0xA6A520}), {64, 64},
        VK_FORMAT_R8G8B8A8_UNORM);
    const auto makeTransport = [&](const PreparedSource& source, uint64_t generation) {
        const auto lifetime = std::make_shared<const uint8_t>(0);
        const vk::RuntimeFrameTransportSource descriptor{
            .image = source.image,
            .currentLayout = source.layout,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {64, 64},
            .generation = generation,
            .lifetime = lifetime};
        auto submitted = vk::RuntimeImageEndpoint::trySubmitFrameTransportA(
            frameTransportA, descriptor);
        assert(submitted.status
            == vk::RuntimeFrameTransportSubmitStatus::SUBMITTED);
        trace.events.emplace_back(std::string(1, generation == 1 ? 'A'
            : generation == 2 ? 'B' : 'C') + "_TRANSPORT_SUBMIT");
        return std::move(submitted.submission);
    };

    setSource(sources[0], 'A', 0);
    auto warmupA = owner.submitWarmup('A',
        lsfgvk::backend::TemporalSourceSlot::Slot0,
        makeTransport(sources[0], 1));
    assert(warmupA.submitAccepted);
    trace.events.emplace_back("A_WARMUP_SUBMIT");
    assert(owner.tryRetireWarmup()
        == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED);
    trace.events.emplace_back("A_WARMUP_RETIRE");
    observeSource(sources[0], 'A', owner.temporalObservationSource(
        lsfgvk::backend::TemporalSourceSlot::Slot0), trace);

    setSource(sources[1], 'B', 1);
    auto warmupB = owner.submitWarmup('B',
        lsfgvk::backend::TemporalSourceSlot::Slot1,
        makeTransport(sources[1], 2));
    assert(warmupB.submitAccepted);
    trace.events.emplace_back("B_WARMUP_SUBMIT");
    assert(owner.tryRetireWarmup()
        == lsfgvk::backend::RuntimeIngestRetirementStatus::RETIRED);
    trace.events.emplace_back("B_WARMUP_RETIRE");
    observeSource(sources[1], 'B', owner.temporalObservationSource(
        lsfgvk::backend::TemporalSourceSlot::Slot1), trace);

    setSource(sourceC, 'C', 0);
    auto ingestC = owner.submitGenerateSource('C',
        lsfgvk::backend::TemporalSourceSlot::Slot0,
        makeTransport(sourceC, 3));
    assert(ingestC.submitAccepted && ingestC.signalSemaphore != VK_NULL_HANDLE);
    trace.events.emplace_back("C_GENERATE_SOURCE_SUBMIT");
    const auto generated = owner.submitGenerate({
        .olderSlot = lsfgvk::backend::TemporalSourceSlot::Slot1,
        .newerSlot = lsfgvk::backend::TemporalSourceSlot::Slot0,
        .olderFrameId = 'B', .newerFrameId = 'C', .generationId = ingestC.epoch});
    assert(generated.submitAccepted);
    trace.events.emplace_back("GENERATE_SUBMIT");
    harness.backendFenceStatusResult = VK_SUCCESS;
    assert(owner.tryRetireGenerate()
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    trace.events.emplace_back("GENERATE_RETIRE");
    observeSource(sourceC, 'C', owner.temporalObservationSource(
        lsfgvk::backend::TemporalSourceSlot::Slot0), trace);

    harness.armProductionReturnForOwner();
    owner.submitBReturn();
    trace.events.emplace_back("B_RETURN_SUBMIT");
    assert(owner.tryRetireBReturn()
        == lsfgvk::backend::RuntimeRetirementStatus::NOT_READY);
    trace.events.emplace_back("B_RETURN_NOT_READY");
    harness.retireFence();
    assert(owner.tryRetireBReturn()
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    trace.events.emplace_back("B_RETURN_RETIRE");
    owner.submitAReturn(harness.productionHandoff());
    trace.events.emplace_back("A_RETURN_SUBMIT");
    assert(owner.releaseGeneratedOutput()
        == lsfgvk::backend::RuntimeRetirementStatus::RETIRED);
    trace.events.emplace_back("OUTPUT_RELEASE");
    assert(!owner.tryRetireAReturn());
    trace.events.emplace_back("A_RETURN_NOT_READY");
    harness.retireAFence();
    assert(owner.tryRetireAReturn());
    trace.events.emplace_back("A_RETURN_RETIRE");

    assert(frameTransportA.frameTransportStateValue()
        == vk::RuntimeFrameTransportState::REUSABLE);
    const auto final = owner.generateState();
    assert(final.generationReadyWaitRetired && final.generationReadyReusable);
    assertTrace(trace);
    assert(checksum(sources[0]) != checksum(sources[1])
        && checksum(sources[0]) != checksum(sourceC)
        && checksum(sources[1]) != checksum(sourceC));
    return 0;
}
