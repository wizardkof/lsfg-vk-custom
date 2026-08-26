/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "d3b3_production_core_owner.hpp"
#include "lsfg-vk-backend/lsfgvk.hpp"
#include "lsfg-vk-common/fnv1a.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/vulkan/buffer.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/external_semaphore_sync.hpp"
#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/vulkan/runtime_image_observation.hpp"
#include "lsfg-vk-common/vulkan/runtime_device_pair.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "runtime_dma_buf_backing.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

namespace backend = lsfgvk::backend;

constexpr VkExtent2D kExtent{64, 64};
constexpr VkFormat kFormat = VK_FORMAT_B8G8R8A8_UNORM;
constexpr size_t kBytes = static_cast<size_t>(kExtent.width) * kExtent.height * 4U;

struct PreparedSource {
    std::unique_ptr<vk::Image> image;
    std::shared_ptr<const uint8_t> lifetime{std::make_shared<const uint8_t>(0)};
    std::array<uint8_t, 4> texel{};
};

struct PollCounters {
    uint64_t warmup{};
    uint64_t generate{};
    uint64_t bReturn{};
    uint64_t aReturn{};
    uint64_t observation{};
    uint64_t sourcePrep{};
};

[[nodiscard]] std::filesystem::path shaderPath() {
    const char* value = std::getenv("LSFGVK_DLL_PATH");
    if (!value || *value == '\0')
        throw std::runtime_error(
            "A6A5H requires LSFGVK_DLL_PATH pointing to Lossless.dll");
    return std::filesystem::path(value);
}

[[nodiscard]] VkPhysicalDevice selectVendor(
        const vk::VulkanInstanceFuncs& funcs,
        const std::vector<VkPhysicalDevice>& devices,
        uint32_t vendor, const char* role) {
    for (const auto device : devices) {
        const auto identity = vk::getPhysicalDeviceIdentity(funcs, device);
        if (identity.vendorId == vendor)
            return device;
    }
    throw std::runtime_error(std::string("A6A5H ") + role
        + " device vendor was not enumerated");
}

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw ls::vulkan_error(result, operation);
}

void prepareSource(vk::Vulkan& renderVk, PreparedSource& source,
        PollCounters& polls, char label) {
    std::vector<uint8_t> bytes(kBytes);
    for (size_t i = 0; i < bytes.size(); i += 4)
        std::copy(source.texel.begin(), source.texel.end(), bytes.begin() + i);
    vk::Buffer staging(renderVk, bytes.data(), bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vk::CommandBuffer command(renderVk);
    command.begin(renderVk);
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, source.image->handle(), range};
    command.insertBarriers(renderVk, {toDst}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy copy{0, 0, 0,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
        {kExtent.width, kExtent.height, 1}};
    renderVk.df().CmdCopyBufferToImage(command.handle(), staging.handle(),
        source.image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    const VkImageMemoryBarrier toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, source.image->handle(), range};
    command.insertBarriers(renderVk, {toSrc}, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    command.end(renderVk);
    vk::Fence fence(renderVk);
    command.submit(renderVk, renderVk.queue(), {}, VK_NULL_HANDLE, 0, {}, VK_NULL_HANDLE,
        0, fence.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT);
    while (true) {
        ++polls.sourcePrep;
        const auto result = renderVk.df().GetFenceStatus(renderVk.dev(), fence.handle());
        if (result == VK_SUCCESS)
            break;
        if (result != VK_NOT_READY)
            check(result, "A6A5H source preparation fence status");
        std::this_thread::yield();
    }
    std::cerr << "A6A5H PREPARE " << label
        << " UNDEFINED->TRANSFER_DST_OPTIMAL->PATTERN->TRANSFER_SRC_OPTIMAL PASS\n";
}

template<typename Submit, typename Retire>
void pollUntilRetired(Submit submit, Retire retire, uint64_t& polls,
        const char* label) {
    static constexpr uint64_t kMaxPolls = 10'000'000;
    submit();
    while (true) {
        ++polls;
        const auto status = retire();
        using Status = std::decay_t<decltype(status)>;
        if constexpr (std::is_same_v<Status, backend::RuntimeIngestRetirementStatus>) {
            if (status == backend::RuntimeIngestRetirementStatus::RETIRED)
                break;
        } else {
            if (status == backend::RuntimeRetirementStatus::RETIRED)
                break;
            if (status == backend::RuntimeRetirementStatus::DEVICE_LOST
                    || status == backend::RuntimeRetirementStatus::FAILED)
                throw std::runtime_error(std::string("A6A5H ") + label + " failed");
        }
        if (polls >= kMaxPolls)
            throw std::runtime_error(std::string("A6A5H ") + label
                + " did not retire within nonblocking poll budget");
        std::this_thread::yield();
    }
}

vk::RuntimeImageObservation observe(vk::RuntimeImageObservationSession& session,
        const vk::RuntimeImageObservationDescriptor& descriptor,
        PollCounters& polls, const char* label) {
    const auto submitted = session.submit(descriptor);
    if (submitted.status != vk::RuntimeImageObservationSubmitStatus::SUBMITTED)
        throw std::runtime_error(std::string("A6A5H observation unexpectedly blocked: ") + label);
    while (true) {
        ++polls.observation;
        const auto result = session.tryRetire(submitted.ticket);
        if (result.status == vk::RuntimeImageObservationRetireStatus::RETIRED
                && result.observation.has_value()) {
            std::cerr << "A6A5H OBSERVE " << label
                << " bytes=" << result.observation->byteCount
                << " nonzero=" << result.observation->nonzeroByteCount
                << " checksum=0x" << std::hex << result.observation->checksum
                << std::dec << " PASS\n";
            return *result.observation;
        }
        std::this_thread::yield();
    }
}

vk::RuntimeFrameTransportSubmission submitTransport(
        vk::RuntimeImageEndpoint& imageA, const PreparedSource& source,
        uint64_t generation) {
    const vk::RuntimeFrameTransportSource descriptor{
        .image = source.image->handle(),
        .currentLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .format = kFormat,
        .extent = kExtent,
        .generation = generation,
        .lifetime = source.lifetime};
    auto result = vk::RuntimeImageEndpoint::trySubmitFrameTransportA(
        imageA, descriptor);
    if (result.status != vk::RuntimeFrameTransportSubmitStatus::SUBMITTED)
        throw std::runtime_error("A6A5H frame transport unexpectedly blocked");
    return std::move(result.submission);
}

} // namespace

int main() {
    try {
        const auto selectA = [](const vk::VulkanInstanceFuncs& funcs,
                const std::vector<VkPhysicalDevice>& devices) {
            return selectVendor(funcs, devices, 0x1002U, "GPU A (AMD)");
        };
        vk::Vulkan renderVk("lsfg-vk-a6a5h-render", vk::version{1, 0, 0},
            "lsfg-vk-a6a5h", vk::version{1, 0, 0}, selectA, false);
        const auto renderIdentity = vk::getPhysicalDeviceIdentity(
            renderVk.fi(), renderVk.physdev());

        const backend::DevicePicker selectB = [](const vk::PhysicalDeviceIdentity& identity) {
            return identity.vendorId == 0x10deU;
        };
        backend::Instance backend(selectB, shaderPath(), false);
        const auto pair = vk::bindRuntimeDevicePair(
            renderIdentity, backend.deviceIdentity(), std::optional<std::string>{"generation"});
        if (!pair || !pair->crossDevice())
            throw std::runtime_error("A6A5H did not bind a cross-device AMD->NVIDIA pair");
        std::cerr << "A6A5H DEVICES A=" << pair->render.identity.name
            << " B=" << pair->generation.identity.name << " PASS\n";

        auto renderEndpoint = vk::makeRuntimeExchangeEndpoint(renderVk);
        auto generationEndpoint = backend.runtimeExchangeEndpoint();
        auto controlBacking = lsfgvk::layer::RuntimeDmaBufBacking::create(renderIdentity);
        const vk::RuntimeExchangeChannelInfo controlInfo{
            .logicalSize = 64 * 1024,
            .backingSize = controlBacking.size(),
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        auto channel = vk::createRuntimeExchangeChannel(*pair,
            renderEndpoint, generationEndpoint,
            controlBacking.duplicateFd(), controlBacking.duplicateFd(), controlInfo);

        auto imageBackingStorage = lsfgvk::layer::RuntimeDmaBufBacking::createImage(
            renderIdentity, kExtent);
        const vk::RuntimeImageBackingInfo imageBacking{
            .extent = kExtent,
            .format = kFormat,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .backingSize = imageBackingStorage.size(),
            .fourcc = imageBackingStorage.fourcc(),
            .modifier = imageBackingStorage.modifier(),
            .planeCount = imageBackingStorage.planeCount(),
            .plane = VkSubresourceLayout{.offset = imageBackingStorage.offset(),
                .size = 0, .rowPitch = imageBackingStorage.stride(),
                .arrayPitch = 0, .depthPitch = 0}};
        auto imageA = vk::createRuntimeImageEndpoint(
            renderEndpoint, imageBackingStorage.duplicatePlaneFd(0), imageBacking);
        auto imageB = vk::createRuntimeImageEndpoint(
            generationEndpoint, imageBackingStorage.duplicatePlaneFd(0), imageBacking);
        imageA = vk::RuntimeImageEndpoint::createExecutionResources(std::move(imageA), 2);
        imageB = vk::RuntimeImageEndpoint::createExecutionResources(std::move(imageB), 1);
        std::cerr << "A6A5H exchangeFrameA/B DMA-BUF import and resources PASS\n";

        std::array<PreparedSource, 3> sources;
        for (auto& source : sources)
            source.image = std::make_unique<vk::Image>(renderVk, kExtent, kFormat,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        sources[0].texel = {0x11, 0x22, 0x33, 0xff};
        sources[1].texel = {0x44, 0x55, 0x66, 0xff};
        sources[2].texel = {0x77, 0x88, 0x99, 0xff};
        PollCounters polls;
        for (size_t i = 0; i < sources.size(); ++i)
            prepareSource(renderVk, sources[i], polls, static_cast<char>('A' + i));

        auto observation = vk::createRuntimeImageObservationSession(
            generationEndpoint, {.slotCount = 1, .maxByteCount = kBytes});
        lsfgvk::layer::D3B3ProductionCoreOwner owner({
            .backend = &backend,
            .devicePair = &*pair,
            .exchangeChannel = &channel,
            .generationEndpoint = generationEndpoint,
            .renderEndpoint = renderEndpoint,
            .extent = kExtent,
            .format = kFormat,
            .modifier = imageBacking.modifier,
            .flow = 1.0F,
            .performanceMode = false,
            .sourceImage = [&](lsfgvk::backend::TemporalSourceSlot) {
                return imageB.image();
            }});
        if (!owner.structurallyReady())
            throw std::runtime_error("A6A5H production owner is not structurally ready");

        auto warmupA = owner.submitWarmup(1,
            lsfgvk::backend::TemporalSourceSlot::Slot0,
            submitTransport(imageA, sources[0], 1));
        if (!warmupA.submitAccepted)
            throw std::runtime_error("A6A5H A warmup submit rejected");
        pollUntilRetired([] {}, [&] { return owner.tryRetireWarmup(); }, polls.warmup,
            "A warmup");
        const auto observedA = observe(observation, owner.temporalObservationSource(
            lsfgvk::backend::TemporalSourceSlot::Slot0), polls, "A");

        auto warmupB = owner.submitWarmup(2,
            lsfgvk::backend::TemporalSourceSlot::Slot1,
            submitTransport(imageA, sources[1], 2));
        if (!warmupB.submitAccepted)
            throw std::runtime_error("A6A5H B warmup submit rejected");
        pollUntilRetired([] {}, [&] { return owner.tryRetireWarmup(); }, polls.warmup,
            "B warmup");
        const auto observedB = observe(observation, owner.temporalObservationSource(
            lsfgvk::backend::TemporalSourceSlot::Slot1), polls, "B");

        auto ingestC = owner.submitGenerateSource(3,
            lsfgvk::backend::TemporalSourceSlot::Slot0,
            submitTransport(imageA, sources[2], 3));
        if (!ingestC.submitAccepted)
            throw std::runtime_error("A6A5H C ingest submit rejected");
        const auto generated = owner.submitGenerate({
            .olderSlot = lsfgvk::backend::TemporalSourceSlot::Slot1,
            .newerSlot = lsfgvk::backend::TemporalSourceSlot::Slot0,
            .olderFrameId = 2, .newerFrameId = 3,
            .generationId = ingestC.epoch});
        if (!generated.submitAccepted)
            throw std::runtime_error("A6A5H Generate submit rejected");
        pollUntilRetired([] {}, [&] { return owner.tryRetireGenerate(); }, polls.generate,
            "Generate");
        const auto observedC = observe(observation, owner.temporalObservationSource(
            lsfgvk::backend::TemporalSourceSlot::Slot0), polls, "C");
        const auto observedOutput = observe(observation, owner.generatedObservationSource(),
            polls, "GENERATED_OUTPUT");

        if (observedA.checksum == 0 || observedA.checksum == observedB.checksum
                || observedB.checksum == observedC.checksum
                || observedC.checksum == observedOutput.checksum)
            throw std::runtime_error("A6A5H deterministic A/B/C observation distinction failed");

        owner.submitBReturn();
        pollUntilRetired([] {}, [&] { return owner.tryRetireBReturn(); }, polls.bReturn,
            "B-return");
        auto returnedSignal = vk::createExportableSyncFdSemaphore(renderEndpoint.semaphoreDevice);
        owner.submitAReturn({renderEndpoint.queueFamilyIndex, returnedSignal.handle()});
        if (owner.releaseGeneratedOutput() != lsfgvk::backend::RuntimeRetirementStatus::RETIRED)
            throw std::runtime_error("A6A5H generated output release did not retire");
        while (!owner.tryRetireAReturn()) {
            ++polls.aReturn;
            std::this_thread::yield();
        }

        const auto finalIngest = owner.ingestState();
        const auto finalGenerate = owner.generateState();
        if (!finalIngest.ingestReadyReusable || !finalGenerate.generationReadyReusable
                || imageA.frameTransportStateValue() != vk::RuntimeFrameTransportState::REUSABLE)
            throw std::runtime_error("A6A5H final reusable-state contract failed");
        std::cerr << "A6A5H REAL A/B/C checksum PASS\n"
            << "A6A5H B,C GENERATE PASS\nA6A5H RETURN PASS\n"
            << "A6A5H FD/OWNERSHIP first-use->reusable PASS\n"
            << "A6A5H HOST_WAIT production execution NONE\n"
            << "A6A5H POLLS sourcePrep=" << polls.sourcePrep
            << " warmup=" << polls.warmup << " generate=" << polls.generate
            << " observation=" << polls.observation << " bReturn=" << polls.bReturn
            << " aReturn=" << polls.aReturn << "\n"
            << "DG2X_P4C_D3B3_A6A5H_REAL_GENERIC_TRANSPORT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A6A5H HARDWARE RUNNER BLOCKED: " << error.what() << "\n";
        return 2;
    }
}
