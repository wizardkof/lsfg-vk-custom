#pragma once

#include "lsfg-vk-backend/lsfgvk.hpp"
#include "generated_output_return_diagnostic.hpp"
#include "lsfg-vk-common/vulkan/runtime_exchange_channel.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "lsfg-vk-common/vulkan/vulkan_native_external_image_backing.hpp"

#include <cstring>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

namespace lsfgvk::test {

struct ObservedQueueSubmit {
    uint32_t calls{};
    VkQueue queue{};
    uint32_t submitCount{};
    uint32_t waitCount{};
    VkSemaphore wait{};
    VkPipelineStageFlags waitStage{};
    uint32_t commandCount{};
    VkCommandBuffer command{};
    uint32_t signalCount{};
    VkSemaphore signal{};
    VkFence fence{};
};

struct ObservedImageCreate {
    VkDevice device{};
    VkFormat format{};
    VkExtent3D extent{};
    VkImageUsageFlags usage{};
    VkImageTiling tiling{};
    VkExternalMemoryHandleTypeFlags externalHandleTypes{};
    uint64_t modifier{};
    uint32_t modifierPlaneCount{};
    VkSubresourceLayout plane{};
    bool explicitModifier{};
    VkImage image{};
    VkResult result{VK_SUCCESS};
};

struct ObservedMemoryRequirements {
    uint32_t calls{};
    VkImage image{};
    VkDeviceSize size{};
    VkDeviceSize alignment{};
    uint32_t memoryTypeBits{};
};

struct ObservedMemoryAllocation {
    uint32_t calls{};
    VkDeviceSize size{};
    uint32_t memoryTypeIndex{};
    VkExternalMemoryHandleTypeFlags exportHandleTypes{};
    bool dedicated{};
    VkDeviceMemory memory{};
    VkResult result{VK_SUCCESS};
};

struct ObservedImageBind {
    uint32_t calls{};
    VkImage image{};
    VkDeviceMemory memory{};
    VkDeviceSize offset{};
    VkResult result{VK_SUCCESS};
};

struct ObservedCommandPool {
    uint32_t createCalls{};
    uint32_t destroyCalls{};
    uint32_t queueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
    VkCommandPoolCreateFlags flags{};
    VkCommandPool pool{};
};

struct ObservedCommandAllocation {
    uint32_t calls{};
    uint32_t freeCalls{};
    uint32_t freedCount{};
    VkCommandPool pool{};
    VkCommandBufferLevel level{};
    uint32_t count{};
    VkCommandBuffer command{};
};

struct ObservedCommandBeginEnd {
    uint32_t beginCalls{};
    uint32_t endCalls{};
    VkCommandBuffer beginCommand{};
    VkCommandBuffer endCommand{};
    VkCommandBufferUsageFlags beginFlags{};
};

struct ObservedPipelineBarrier {
    VkCommandBuffer command{};
    VkPipelineStageFlags sourceStage{};
    VkPipelineStageFlags destinationStage{};
    VkDependencyFlags dependencyFlags{};
    uint32_t memoryBarrierCount{};
    uint32_t bufferBarrierCount{};
    uint32_t imageBarrierCount{};
    VkBufferMemoryBarrier bufferBarrier{};
    VkImageMemoryBarrier imageBarrier{};
};

struct ObservedImageCopy {
    uint32_t calls{};
    VkCommandBuffer command{};
    VkImage source{};
    VkImageLayout sourceLayout{};
    VkImage destination{};
    VkImageLayout destinationLayout{};
    uint32_t regionCount{};
    VkImageCopy region{};
};

struct ObservedImageToBufferCopy {
    uint32_t calls{};
    VkCommandBuffer command{};
    VkImage source{};
    VkImageLayout sourceLayout{};
    VkBuffer destination{};
    uint32_t regionCount{};
    VkBufferImageCopy region{};
};

struct ObservedSemaphoreImport {
    uint32_t calls{};
    VkDevice device{};
    VkSemaphore semaphore{};
    VkSemaphoreImportFlags flags{};
    VkExternalSemaphoreHandleTypeFlagBits handleType{};
    int fd{-2};
    VkResult result{VK_SUCCESS};
};

enum class ShadowHarnessFailurePoint {
    NONE,
    CONSUMER_IMAGE_CREATE,
    RETURN_IMAGE_CREATE,
    MEMORY_ALLOCATION,
    IMAGE_BIND,
    COMMAND_POOL_CREATE,
    COMMAND_BUFFER_ALLOCATION,
    BEGIN_COMMAND_BUFFER,
    END_COMMAND_BUFFER
};

enum class ShadowAHarnessFailurePoint {
    NONE,
    IMAGE_CREATE,
    MEMORY_IMPORT,
    STAGING_BUFFER_CREATE,
    STAGING_MEMORY_ALLOCATION,
    COMMAND_POOL_CREATE,
    COMMAND_BUFFER_ALLOCATION,
    BEGIN_COMMAND_BUFFER,
    END_COMMAND_BUFFER,
    FENCE_CREATE,
    SEMAPHORE_CREATE,
    SEMAPHORE_IMPORT
};

struct StagedShadowBReturnView {
    backend::RuntimeTemporalPairIdentity pair{};
    backend::RuntimeGenerationId generation{};
    VkSemaphore generationReady{};
    VkSemaphore futureWaitSemaphore{};
    VkImage generatedImage{};
    VkImage futureCopySource{};
    VkExtent2D extent{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    VkDevice device{};
    VkQueue queue{};
    VkQueue futureSubmitQueue{};
    uint32_t queueFamily{VK_QUEUE_FAMILY_IGNORED};
    uint32_t futureCommandPoolFamily{VK_QUEUE_FAMILY_IGNORED};
    VkFence sourceReadRetirementFence{};
    VkImage futureReturnBackingImage{};
    VkCommandPool futureReturnCommandPool{};
    VkCommandBuffer futureReturnCommand{};
    VkSemaphore futureExportSignal{};
    VkFence futureShadowRealFence{};
    bool sourceReadAuthorityPresent{};
    bool outputAuthorityLive{};
    bool transportConsumed{};
    bool retirementAuthorityIssued{};
    bool operationRetired{};
    bool generationReadySignalOutstanding{};
    bool generationReadyWaitSubmitted{};
    bool generationReadyWaitRetired{};
    bool generationReadyReusable{};
};

class ShadowBReturnExecutionHarness {
public:
    ShadowBReturnExecutionHarness() : previousActive(active) { active = this; }
    ShadowBReturnExecutionHarness(const ShadowBReturnExecutionHarness&) = delete;
    ShadowBReturnExecutionHarness& operator=(const ShadowBReturnExecutionHarness&) = delete;
    ~ShadowBReturnExecutionHarness() {
        staged.reset();
        if (active == this) active = previousActive;
    }

    [[nodiscard]] vk::RuntimeExchangeEndpoint endpoint() const {
        return stagedEndpoint.has_value() ? *stagedEndpoint : generationEndpoint();
    }

    [[nodiscard]] std::unique_ptr<backend::Instance> createB0B3Instance() {
        if (fullBackendMode)
            throw std::logic_error("B0B3 test instance already created");
        fullBackendMode = true;
        return std::make_unique<backend::Instance>(instance, device,
            generationPhysicalDevice, instanceFunctionTable(), deviceFunctionTable(),
            shaderResources());
    }

    [[nodiscard]] vk::PhysicalDeviceIdentity generationIdentity() const {
        vk::PhysicalDeviceIdentity result;
        result.name = "generation";
        result.deviceUuid.back() = 0xB3;
        result.driverUuid.back() = 0xB4;
        return result;
    }

    [[nodiscard]] vk::PhysicalDeviceIdentity renderIdentity() const {
        vk::PhysicalDeviceIdentity result;
        result.name = "render";
        result.deviceUuid.back() = 0xA3;
        result.driverUuid.back() = 0xA4;
        return result;
    }

    [[nodiscard]] vk::RuntimeDevicePair devicePair() const {
        auto pair = vk::bindRuntimeDevicePair(
            renderIdentity(), generationIdentity(), std::optional<std::string>{"generation"});
        if (!pair) throw std::logic_error("test device pair binding failed");
        return *pair;
    }

    [[nodiscard]] vk::RuntimeExchangeEndpoint generationEndpoint() const {
        auto result = baseImageEndpoint(device, generationPhysicalDevice);
        result.identity = generationIdentity();
        result.semaphoreDevice = {device, {
            createSemaphore, destroySemaphore, getSemaphoreFd, nullptr}};
        result.queue = queue;
        result.queueFamilyIndex = generationQueueFamily;
        result.QueueSubmit = queueSubmit;
        result.CreateFence = createFence;
        result.DestroyFence = destroyFence;
        result.WaitForFences = waitForFences;
        result.ResetFences = resetFences;
        result.DeviceWaitIdle = deviceWaitIdle;
        result.CreateCommandPool = createCommandPool;
        result.DestroyCommandPool = destroyCommandPool;
        result.AllocateCommandBuffers = allocateCommandBuffers;
        result.FreeCommandBuffers = freeCommandBuffers;
        result.BeginCommandBuffer = beginCommandBuffer;
        result.EndCommandBuffer = endCommandBuffer;
        result.CmdPipelineBarrier = cmdPipelineBarrier;
        result.CmdCopyImage = cmdCopyImage;
        result.GetImageMemoryRequirements2 = getImageMemoryRequirements2;
        result.AllocateMemory = allocateMemory;
        result.FreeMemory = freeMemory;
        result.BindImageMemory = bindImageMemory;
        result.GetMemoryFdKHR = getMemoryFd;
        return result;
    }

    [[nodiscard]] vk::RuntimeExchangeEndpoint renderEndpoint() const {
        auto result = baseImageEndpoint(renderDevice, renderPhysicalDevice);
        result.identity = renderIdentity();
        result.bufferDevice.memoryProperties.memoryTypeCount = 2;
        result.bufferDevice.memoryProperties.memoryTypes[0].propertyFlags =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        result.bufferDevice.memoryProperties.memoryTypes[1].propertyFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        result.bufferDevice.funcs = {
            createBuffer, destroyBuffer, getBufferMemoryRequirements,
            getMemoryFdProperties, allocateMemory, freeMemory, bindBufferMemory};
        result.semaphoreDevice = {renderDevice, {
            createSemaphore, destroySemaphore, nullptr, importSemaphoreFd}};
        result.queue = renderQueue;
        result.queueFamilyIndex = renderQueueFamily;
        result.QueueSubmit = queueSubmit;
        result.CreateFence = createFence;
        result.DestroyFence = destroyFence;
        result.WaitForFences = waitForFences;
        result.ResetFences = resetFences;
        result.DeviceWaitIdle = deviceWaitIdle;
        result.CreateCommandPool = createCommandPool;
        result.DestroyCommandPool = destroyCommandPool;
        result.AllocateCommandBuffers = allocateCommandBuffers;
        result.FreeCommandBuffers = freeCommandBuffers;
        result.BeginCommandBuffer = beginCommandBuffer;
        result.EndCommandBuffer = endCommandBuffer;
        result.CmdPipelineBarrier = cmdPipelineBarrier;
        result.MapMemory = mapMemory;
        result.UnmapMemory = unmapMemory;
        result.InvalidateMappedMemoryRanges = invalidateMappedMemoryRanges;
        result.GetImageMemoryRequirements2 = getImageMemoryRequirements2;
        result.BindImageMemory = bindImageMemory;
        result.GetMemoryFdPropertiesKHR = getMemoryFdProperties;
        result.AllocateMemory = allocateMemory;
        result.FreeMemory = freeMemory;
        result.CreateBuffer = createBuffer;
        result.DestroyBuffer = destroyBuffer;
        result.GetBufferMemoryRequirements = getBufferMemoryRequirements;
        result.BindBufferMemory = bindBufferMemory;
        result.CmdCopyImageToBuffer = cmdCopyImageToBuffer;
        return result;
    }

    [[nodiscard]] bool productionResourcePfnInventoryComplete() const {
        const auto generation = generationEndpoint();
        const auto render = renderEndpoint();
        return generation.CreateFence && generation.DestroyFence
            && generation.CreateImage && generation.DestroyImage
            && generation.GetPhysicalDeviceFormatProperties2
            && generation.GetPhysicalDeviceImageFormatProperties2
            && generation.GetImageDrmFormatModifierPropertiesEXT
            && generation.GetImageSubresourceLayout
            && generation.GetImageMemoryRequirements2
            && generation.AllocateMemory && generation.FreeMemory
            && generation.BindImageMemory && generation.GetMemoryFdKHR
            && generation.CreateCommandPool && generation.DestroyCommandPool
            && generation.AllocateCommandBuffers && generation.FreeCommandBuffers
            && generation.BeginCommandBuffer && generation.CmdPipelineBarrier
            && generation.CmdCopyImage && generation.EndCommandBuffer
            && generation.semaphoreDevice.funcs.CreateSemaphore
            && generation.semaphoreDevice.funcs.DestroySemaphore
            && generation.semaphoreDevice.funcs.GetSemaphoreFdKHR
            && generation.QueueSubmit && generation.DeviceWaitIdle
            && render.CreateImage && render.DestroyImage
            && render.GetPhysicalDeviceFormatProperties2
            && render.GetPhysicalDeviceImageFormatProperties2
            && render.GetImageDrmFormatModifierPropertiesEXT
            && render.GetImageSubresourceLayout
            && render.GetImageMemoryRequirements2 && render.AllocateMemory
            && render.FreeMemory && render.BindImageMemory
            && render.GetMemoryFdPropertiesKHR
            && render.CreateBuffer && render.DestroyBuffer
            && render.GetBufferMemoryRequirements && render.BindBufferMemory
            && render.MapMemory && render.UnmapMemory
            && render.CreateCommandPool && render.DestroyCommandPool
            && render.AllocateCommandBuffers && render.FreeCommandBuffers
            && render.BeginCommandBuffer && render.EndCommandBuffer
            && render.CmdPipelineBarrier && render.CmdCopyImageToBuffer
            && render.QueueSubmit && render.CreateFence && render.DestroyFence
            && render.WaitForFences
            && render.semaphoreDevice.funcs.CreateSemaphore
            && render.semaphoreDevice.funcs.DestroySemaphore
            && render.semaphoreDevice.funcs.ImportSemaphoreFdKHR;
    }

    // Harness-only resource/recorder self-test. It deliberately calls neither
    // submitGeneratedBReturn() nor QueueSubmit.
    void exerciseResourceAndCommandPath() {
        auto generation = generationEndpoint();
        auto render = renderEndpoint();
        auto backing = vk::VulkanNativeExternalImageBacking::create(
            generation, render, returnExtent, returnFormat);

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        auto cleanupCommands = [&] {
            if (command && pool && generation.FreeCommandBuffers)
                generation.FreeCommandBuffers(device, pool, 1, &command);
            if (pool && generation.DestroyCommandPool)
                generation.DestroyCommandPool(device, pool, nullptr);
            command = VK_NULL_HANDLE;
            pool = VK_NULL_HANDLE;
        };

        try {
            const VkCommandPoolCreateInfo poolInfo{
                VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0,
                generationQueueFamily};
            auto result = generation.CreateCommandPool(
                device, &poolInfo, nullptr, &pool);
            if (result != VK_SUCCESS)
                throw std::runtime_error("harness command-pool creation failed");

            const VkCommandBufferAllocateInfo allocation{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            result = generation.AllocateCommandBuffers(device, &allocation, &command);
            if (result != VK_SUCCESS)
                throw std::runtime_error("harness command-buffer allocation failed");

            const VkCommandBufferBeginInfo begin{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
            result = generation.BeginCommandBuffer(command, &begin);
            if (result != VK_SUCCESS)
                throw std::runtime_error("harness command-buffer begin failed");

            const VkImageSubresourceRange range{
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const VkImageMemoryBarrier toDestination{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                generationQueueFamily, backing.image(), range};
            generation.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &toDestination);

            const VkImageCopy copy{
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                {returnExtent.width, returnExtent.height, 1}};
            generation.CmdCopyImage(command, generatedSourceImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, backing.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

            const VkImageMemoryBarrier release{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, generationQueueFamily,
                VK_QUEUE_FAMILY_FOREIGN_EXT, backing.image(), range};
            generation.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &release);

            result = generation.EndCommandBuffer(command);
            if (result != VK_SUCCESS)
                throw std::runtime_error("harness command-buffer end failed");

            // Exercise the exact backing-memory export PFN without treating the
            // returned test pipe as a real DMA-BUF.
            auto descriptor = backing.exportDescriptor();
            if (!descriptor.dmaBuf)
                throw std::runtime_error("harness backing export returned no test fd");
        } catch (...) {
            cleanupCommands();
            throw;
        }
        cleanupCommands();
    }

    VkResult submitResult{VK_SUCCESS};
    VkResult exportResult{VK_SUCCESS};
    VkResult memoryExportResult{VK_SUCCESS};
    VkResult fenceWaitResult{VK_TIMEOUT};
    VkResult sourceFenceWaitResult{VK_TIMEOUT};
    VkResult aSubmitResult{VK_SUCCESS};
    VkResult aFenceWaitResult{VK_TIMEOUT};
    VkResult aImportResult{VK_SUCCESS};
    ShadowHarnessFailurePoint failurePoint{ShadowHarnessFailurePoint::NONE};
    ShadowAHarnessFailurePoint aFailurePoint{ShadowAHarnessFailurePoint::NONE};

    ObservedQueueSubmit observed{};
    ObservedQueueSubmit aObserved{};
    std::vector<ObservedImageCreate> imageCreates;
    std::vector<VkImage> destroyedImages;
    ObservedMemoryRequirements memoryRequirements{};
    ObservedMemoryAllocation memoryAllocation{};
    ObservedImageBind imageBind{};
    ObservedCommandPool commandPoolObservation{};
    ObservedCommandAllocation commandAllocation{};
    ObservedCommandBeginEnd commandBeginEnd{};
    std::vector<ObservedPipelineBarrier> barriers;
    ObservedImageCopy imageCopy{};
    std::vector<std::string> commandTrace;
    ObservedMemoryRequirements aMemoryRequirements{};
    std::vector<ObservedMemoryAllocation> aMemoryAllocations;
    std::vector<VkDeviceMemory> aFreedMemories;
    ObservedImageBind aImageBind{};
    ObservedCommandPool aCommandPoolObservation{};
    ObservedCommandAllocation aCommandAllocation{};
    ObservedCommandBeginEnd aCommandBeginEnd{};
    std::vector<ObservedPipelineBarrier> aBarriers;
    ObservedImageToBufferCopy aImageToBuffer{};
    ObservedSemaphoreImport aSemaphoreImport{};
    std::vector<std::string> aCommandTrace;

    uint32_t formatPropertyQueries{};
    uint32_t imageFormatPropertyQueries{};
    uint32_t modifierQueries{};
    uint32_t subresourceLayoutQueries{};
    uint32_t freedMemories{};
    uint32_t memoryFdExports{};
    uint32_t semaphoreCreateCalls{};
    uint32_t semaphoreDestroyCalls{};
    uint32_t exportCalls{};
    uint32_t fenceCreateCalls{};
    uint32_t fenceDestroyCalls{};
    uint32_t fenceWaitCalls{};
    uint32_t sourceFenceWaitCalls{};
    uint32_t aFenceWaitCalls{};
    uint32_t deviceIdleCalls{};
    uint32_t productionBReturnCalls{};
    uint32_t productionBReturnExportCalls{};
    uint32_t aPhaseCalls{};
    uint32_t aImportCalls{};
    uint32_t aQueueSubmitCalls{};
    uint32_t foreignReadbackPendingCalls{};
    uint32_t returnedForGraphicsCalls{};
    uint32_t terminalCalls{};
    uint32_t aBufferCreateCalls{};
    uint32_t aBufferDestroyCalls{};
    uint32_t aBufferBindCalls{};
    uint32_t aMapCalls{};
    uint32_t aUnmapCalls{};
    uint32_t aMemoryFdPropertyCalls{};
    uint32_t aSemaphoreCreateCalls{};
    uint32_t aSemaphoreDestroyCalls{};
    uint32_t aFenceCreateCalls{};
    uint32_t aFenceDestroyCalls{};
    int lastMemoryExportFd{-1};
    VkDeviceMemory lastExportedMemory{};
    VkExternalMemoryHandleTypeFlagBits lastMemoryExportHandleType{};
    VkSemaphore lastExportedSemaphore{};
    bool fenceRetired{};
    bool sourceFenceRetired{};
    bool aFenceRetired{};
    bool aExternalMemoryImported{};

    static constexpr uint32_t generationQueueFamily = 3;
    static constexpr uint32_t renderQueueFamily = 7;
    static constexpr uint32_t terminalQueueFamily = 11;
    static constexpr VkDeviceSize returnAllocationSize = 16384;
    static constexpr VkDeviceSize returnAllocationAlignment = 256;
    static constexpr uint32_t returnMemoryTypeBits = 1;

    const VkInstance instance = handle<VkInstance>(0xB300);
    const VkDevice device = handle<VkDevice>(0xB301);
    const VkQueue queue = handle<VkQueue>(0xB302);
    const VkSemaphore generationReady = handle<VkSemaphore>(0xB303);
    const VkSemaphore exportSignal = handle<VkSemaphore>(0xB304);
    const VkCommandBuffer returnCommand = handle<VkCommandBuffer>(0xB305);
    const VkFence returnFence = handle<VkFence>(0xB306);
    const VkImage returnImage = handle<VkImage>(0xB307);
    const VkDeviceMemory returnMemory = handle<VkDeviceMemory>(0xB308);
    const VkCommandPool returnCommandPool = handle<VkCommandPool>(0xB309);
    const VkImage generatedSourceImage = handle<VkImage>(0xB30A);
    const VkPhysicalDevice generationPhysicalDevice = handle<VkPhysicalDevice>(0xB30B);
    const VkDevice renderDevice = handle<VkDevice>(0xA301);
    const VkQueue renderQueue = handle<VkQueue>(0xA302);
    const VkPhysicalDevice renderPhysicalDevice = handle<VkPhysicalDevice>(0xA303);
    const VkImage consumerProbeImage = handle<VkImage>(0xA304);
    const VkImage aImportedImage = handle<VkImage>(0xA305);
    const VkDeviceMemory aPlainProbeMemory = handle<VkDeviceMemory>(0xA306);
    const VkDeviceMemory aImportedMemory = handle<VkDeviceMemory>(0xA307);
    const VkBuffer aStagingBuffer = handle<VkBuffer>(0xA308);
    const VkDeviceMemory aStagingMemory = handle<VkDeviceMemory>(0xA309);
    const VkCommandPool aCommandPool = handle<VkCommandPool>(0xA30A);
    const VkCommandBuffer aCommand = handle<VkCommandBuffer>(0xA30B);
    const VkFence aReturnFence = handle<VkFence>(0xA30C);
    const VkSemaphore aImportedWait = handle<VkSemaphore>(0xA30D);
    const VkSemaphore returnedForGraphics = handle<VkSemaphore>(0xA30E);
    const VkExtent2D returnExtent{8, 8};
    const VkFormat returnFormat{VK_FORMAT_R8G8B8A8_UNORM};
    const VkSubresourceLayout returnPlane{
        0, returnAllocationSize, 256, returnAllocationSize, returnAllocationSize};

    [[nodiscard]] vk::SyncFdPayload exportTestPayload() {
        return vk::exportSyncFd(endpoint().semaphoreDevice, exportSignal);
    }

    void stageProductionPending(backend::RuntimeGenerateDiagnosticPending&& value,
            const backend::RuntimeShadowGenerateSnapshot& producer,
            vk::RuntimeExchangeEndpoint productionEndpoint) {
        if (staged.has_value())
            throw std::logic_error("shadow pending already staged");
        if (!value.valid())
            throw std::invalid_argument("missing B0B3 shadow pending");

        const auto pair = value.temporalPair();
        const auto extent = value.extentValue();
        if (!backend::validTemporalPair(pair)
                || producer.generationId != value.identity()
                || producer.olderFrameId != pair.olderFrameId
                || producer.newerFrameId != pair.newerFrameId
                || producer.olderSlot != pair.olderSlot
                || producer.newerSlot != pair.newerSlot
                || producer.signalSemaphore != value.readinessSemaphore()
                || producer.generatedImage != value.imageHandle()
                || producer.fence != value.sourceReadRetirementFence()
                || !producer.submitAccepted || producer.executionRetired
                || producer.generationReadyReusable
                || producer.bReturnSubmitCount != 0 || producer.aReturnSubmitCount != 0)
            throw std::invalid_argument("B0B3 pending/snapshot identity mismatch");
        if (value.readinessSemaphore() == VK_NULL_HANDLE
                || value.imageHandle() == VK_NULL_HANDLE
                || value.sourceReadRetirementFence() == VK_NULL_HANDLE
                || value.layoutValue() != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                || value.formatValue() == VK_FORMAT_UNDEFINED
                || extent.width == 0 || extent.height == 0
                || value.transportConsumedForTesting()
                || value.retirementAuthorityIssuedForTesting()
                || value.operationRetiredForTesting())
            throw std::invalid_argument("incomplete B0B3 production metadata");
        if (productionEndpoint.bufferDevice.device == VK_NULL_HANDLE
                || productionEndpoint.queue == VK_NULL_HANDLE
                || productionEndpoint.queueFamilyIndex != value.queueFamily()
                || productionEndpoint.QueueSubmit != queueSubmit
                || productionEndpoint.CreateFence != createFence
                || productionEndpoint.CreateCommandPool != createCommandPool
                || productionEndpoint.AllocateCommandBuffers != allocateCommandBuffers
                || productionEndpoint.BeginCommandBuffer != beginCommandBuffer
                || productionEndpoint.EndCommandBuffer != endCommandBuffer
                || productionEndpoint.CmdPipelineBarrier != cmdPipelineBarrier
                || productionEndpoint.CmdCopyImage != cmdCopyImage
                || productionEndpoint.CreateImage != createImage
                || productionEndpoint.DestroyImage != destroyImage
                || productionEndpoint.GetImageMemoryRequirements2
                    != getImageMemoryRequirements2
                || productionEndpoint.AllocateMemory != allocateMemory
                || productionEndpoint.BindImageMemory != bindImageMemory
                || productionEndpoint.semaphoreDevice.funcs.CreateSemaphore
                    != createSemaphore
                || productionEndpoint.semaphoreDevice.funcs.GetSemaphoreFdKHR
                    != getSemaphoreFd)
            throw std::invalid_argument("B0B3 session/harness dispatch identity mismatch");

        StagedShadowBReturnView next{};
        next.pair = pair;
        next.generation = value.identity();
        next.generationReady = value.readinessSemaphore();
        next.futureWaitSemaphore = value.readinessSemaphore();
        next.generatedImage = value.imageHandle();
        next.futureCopySource = value.imageHandle();
        next.extent = extent;
        next.format = value.formatValue();
        next.layout = value.layoutValue();
        next.device = productionEndpoint.bufferDevice.device;
        next.queue = productionEndpoint.queue;
        next.futureSubmitQueue = productionEndpoint.queue;
        next.queueFamily = productionEndpoint.queueFamilyIndex;
        next.futureCommandPoolFamily = productionEndpoint.queueFamilyIndex;
        next.sourceReadRetirementFence = value.sourceReadRetirementFence();
        next.futureReturnBackingImage = returnImage;
        next.futureReturnCommandPool = returnCommandPool;
        next.futureReturnCommand = returnCommand;
        next.futureExportSignal = exportSignal;
        next.futureShadowRealFence = layer::returnSubmissionFence(
            layer::ReturnSubmissionFencePolicy::SHADOW_REAL, returnFence);
        next.sourceReadAuthorityPresent =
            next.sourceReadRetirementFence != VK_NULL_HANDLE;
        next.transportConsumed = value.transportConsumedForTesting();
        next.retirementAuthorityIssued = value.retirementAuthorityIssuedForTesting();
        next.operationRetired = value.operationRetiredForTesting();
        next.outputAuthorityLive = value.valid() && !next.operationRetired;
        next.generationReadySignalOutstanding =
            producer.submitAccepted && !producer.generationReadyReusable;
        next.generationReadyWaitSubmitted = false;
        next.generationReadyWaitRetired = false;
        next.generationReadyReusable = producer.generationReadyReusable;

        staged.emplace(std::move(value));
        stagedEndpoint.emplace(std::move(productionEndpoint));
        stagedProduction.emplace(next);
        returnResourcesArmed = true;
    }
    [[nodiscard]] bool hasStagedPending() const noexcept {
        return staged.has_value() && staged->valid();
    }
    [[nodiscard]] const StagedShadowBReturnView& stagedView() const {
        if (!stagedProduction)
            throw std::logic_error("no staged B0B3 production state");
        return *stagedProduction;
    }
    void validateStagedReturnConfiguration(
            const vk::RuntimeExchangeEndpoint& candidate,
            VkExtent2D candidateExtent, VkFormat candidateFormat) const {
        const auto& view = stagedView();
        if (candidate.bufferDevice.device != view.device
                || candidate.queue != view.queue
                || candidate.queueFamilyIndex != view.queueFamily
                || candidateExtent.width != view.extent.width
                || candidateExtent.height != view.extent.height
                || candidateFormat != view.format)
            throw std::invalid_argument("staged B-return resource identity mismatch");
    }
    [[nodiscard]] backend::RuntimeGenerateDiagnosticPending takeStaged() {
        if (!hasStagedPending())
            throw std::logic_error("no staged shadow pending to move");
        auto result = std::move(*staged);
        staged.reset();
        stagedEndpoint.reset();
        stagedProduction.reset();
        returnResourcesArmed = false;
        return result;
    }

    [[nodiscard]] backend::RuntimeGenerateDiagnosticPending takeStagedForExecution() {
        if (!hasStagedPending() || !stagedEndpoint || !stagedProduction)
            throw std::logic_error("no staged shadow pending to execute");
        auto result = std::move(*staged);
        staged.reset();
        return result;
    }

    [[nodiscard]] layer::RuntimeGeneratedBReturnPending submitProductionBReturn(
            layer::GeneratedOutputReturnDiagnosticSession& returnSession,
            backend::RuntimeGenerateDiagnosticPending&& pending,
            backend::Instance& backendInstance,
            backend::RuntimeGenerateDiagnosticSession& backendSession) {
        ++productionBReturnCalls;
        return returnSession.submitShadowBReturnForTesting(
            std::move(pending), backendInstance, backendSession);
    }

    [[nodiscard]] backend::ReturnedGeneratedOperation completeProductionAReturn(
            layer::GeneratedOutputReturnDiagnosticSession& returnSession,
            layer::RuntimeGeneratedBReturnPending&& pending,
            backend::Instance& backendInstance,
            backend::RuntimeGenerateDiagnosticSession& backendSession) {
        ++aPhaseCalls;
        return returnSession.completeShadowGeneratedReturnOnAForTesting(
            std::move(pending), backendInstance, backendSession,
            vk::RuntimeForeignImageHandoffInfo{
                terminalQueueFamily, returnedForGraphics});
    }

    [[nodiscard]] size_t frontSubmitCount() const noexcept {
        return frontSubmits.size();
    }
    [[nodiscard]] const ObservedQueueSubmit& frontSubmit(size_t index) const {
        return frontSubmits.at(index);
    }
    [[nodiscard]] uint32_t frontFenceWaitCount() const noexcept {
        return backendFenceWaitCalls;
    }
    [[nodiscard]] uint32_t frontImportedFdCount() const noexcept {
        return backendImportedFdCalls;
    }

    void retireFence() noexcept { fenceRetired = true; fenceWaitResult = VK_SUCCESS; }
    void retireSourceFence() noexcept {
        sourceFenceRetired = true;
        sourceFenceWaitResult = VK_SUCCESS;
    }
    void retireAFence() noexcept {
        aFenceRetired = true;
        aFenceWaitResult = VK_SUCCESS;
    }

private:
    template<class T> static T handle(uintptr_t value) {
        if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
        else return static_cast<T>(value);
    }

    [[nodiscard]] vk::RuntimeExchangeEndpoint baseImageEndpoint(
            VkDevice endpointDevice, VkPhysicalDevice physicalDevice) const {
        vk::RuntimeExchangeEndpoint result{};
        result.bufferDevice.device = endpointDevice;
        result.bufferDevice.memoryProperties.memoryTypeCount = 1;
        result.bufferDevice.memoryProperties.memoryTypes[0].propertyFlags =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        result.physicalDevice = physicalDevice;
        result.GetPhysicalDeviceFormatProperties2 = getPhysicalDeviceFormatProperties2;
        result.GetPhysicalDeviceImageFormatProperties2 =
            getPhysicalDeviceImageFormatProperties2;
        result.CreateImage = createImage;
        result.DestroyImage = destroyImage;
        result.GetImageSubresourceLayout = getImageSubresourceLayout;
        result.GetImageDrmFormatModifierPropertiesEXT =
            getImageDrmFormatModifierProperties;
        return result;
    }

    static inline ShadowBReturnExecutionHarness* active{};
    ShadowBReturnExecutionHarness* previousActive{};
    std::optional<backend::RuntimeGenerateDiagnosticPending> staged;
    std::optional<vk::RuntimeExchangeEndpoint> stagedEndpoint;
    std::optional<StagedShadowBReturnView> stagedProduction;
    bool fullBackendMode{};
    bool returnResourcesArmed{};
    uintptr_t nextBackendHandle{0xD000};
    std::vector<ObservedQueueSubmit> frontSubmits;
    std::vector<uint8_t> mappedStorage = std::vector<uint8_t>(1024U * 1024U);
    uint32_t backendFenceWaitCalls{};
    uint32_t backendImportedFdCalls{};

    template<class T> [[nodiscard]] T nextHandle() {
        return handle<T>(nextBackendHandle++);
    }

    [[nodiscard]] static std::unordered_map<uint32_t, std::vector<uint8_t>>
    shaderResources() {
        std::unordered_map<uint32_t, std::vector<uint8_t>> result;
        for (uint32_t id = 255; id <= 279; ++id) {
            result.emplace(98U + id, std::vector<uint8_t>(20));
            result.emplace(121U + id, std::vector<uint8_t>(20));
        }
        return result;
    }

    [[nodiscard]] static ObservedQueueSubmit observeSubmit(VkQueue queue,
            uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
        ObservedQueueSubmit out{};
        out.calls = 1;
        out.queue = queue;
        out.submitCount = count;
        out.fence = fence;
        if (count && submits) {
            const auto& submit = submits[0];
            out.waitCount = submit.waitSemaphoreCount;
            if (out.waitCount) {
                out.wait = submit.pWaitSemaphores[0];
                out.waitStage = submit.pWaitDstStageMask[0];
            }
            out.commandCount = submit.commandBufferCount;
            if (out.commandCount) out.command = submit.pCommandBuffers[0];
            out.signalCount = submit.signalSemaphoreCount;
            if (out.signalCount) out.signal = submit.pSignalSemaphores[0];
        }
        return out;
    }

    [[nodiscard]] static vk::VulkanInstanceFuncs instanceFunctionTable() {
        vk::VulkanInstanceFuncs result{};
        result.EnumeratePhysicalDevices = enumeratePhysicalDevices;
        result.EnumerateDeviceExtensionProperties = enumerateDeviceExtensionProperties;
        result.GetPhysicalDeviceProperties2 = getPhysicalDeviceProperties2;
        result.GetPhysicalDeviceQueueFamilyProperties =
            getPhysicalDeviceQueueFamilyProperties;
        result.GetPhysicalDeviceFeatures2 = getPhysicalDeviceFeatures2;
        result.GetPhysicalDeviceMemoryProperties = getPhysicalDeviceMemoryProperties;
        result.GetDeviceProcAddr = getDeviceProcAddr;
        result.GetPhysicalDeviceImageFormatProperties2 =
            getPhysicalDeviceImageFormatProperties2;
        result.GetPhysicalDeviceFormatProperties2 = getPhysicalDeviceFormatProperties2;
        return result;
    }

    [[nodiscard]] static vk::VulkanDeviceFuncs deviceFunctionTable() {
        vk::VulkanDeviceFuncs result{};
        result.GetDeviceQueue = getDeviceQueue;
        result.DeviceWaitIdle = deviceWaitIdle;
        result.CreateCommandPool = createCommandPool;
        result.DestroyCommandPool = destroyCommandPool;
        result.CreateDescriptorPool = createDescriptorPool;
        result.DestroyDescriptorPool = destroyDescriptorPool;
        result.CreateBuffer = createBuffer;
        result.DestroyBuffer = destroyBuffer;
        result.GetBufferMemoryRequirements = getBufferMemoryRequirements;
        result.AllocateMemory = allocateMemory;
        result.FreeMemory = freeMemory;
        result.BindBufferMemory = bindBufferMemory;
        result.MapMemory = mapMemory;
        result.UnmapMemory = unmapMemory;
        result.InvalidateMappedMemoryRanges = invalidateMappedMemoryRanges;
        result.AllocateCommandBuffers = allocateCommandBuffers;
        result.FreeCommandBuffers = freeCommandBuffers;
        result.BeginCommandBuffer = beginCommandBuffer;
        result.EndCommandBuffer = endCommandBuffer;
        result.CmdPipelineBarrier = cmdPipelineBarrier;
        result.CmdFillBuffer = cmdFillBuffer;
        result.CmdCopyBuffer = cmdCopyBuffer;
        result.CmdBlitImage = cmdBlitImage;
        result.CmdClearColorImage = cmdClearColorImage;
        result.CmdBindPipeline = cmdBindPipeline;
        result.CmdBindDescriptorSets = cmdBindDescriptorSets;
        result.CmdDispatch = cmdDispatch;
        result.CmdCopyBufferToImage = cmdCopyBufferToImage;
        result.CmdCopyImageToBuffer = cmdCopyImageToBuffer;
        result.QueueSubmit = queueSubmit;
        result.AllocateDescriptorSets = allocateDescriptorSets;
        result.FreeDescriptorSets = freeDescriptorSets;
        result.UpdateDescriptorSets = updateDescriptorSets;
        result.CreateFence = createFence;
        result.DestroyFence = destroyFence;
        result.ResetFences = resetFences;
        result.WaitForFences = waitForFences;
        result.CreateImage = createImage;
        result.DestroyImage = destroyImage;
        result.GetImageMemoryRequirements = getImageMemoryRequirements;
        result.GetImageMemoryRequirements2 = getImageMemoryRequirements2;
        result.BindImageMemory = bindImageMemory;
        result.CreateImageView = createImageView;
        result.DestroyImageView = destroyImageView;
        result.CreateSampler = createSampler;
        result.DestroySampler = destroySampler;
        result.CreateSemaphore = createSemaphore;
        result.DestroySemaphore = destroySemaphore;
        result.CreateShaderModule = createShaderModule;
        result.DestroyShaderModule = destroyShaderModule;
        result.CreateDescriptorSetLayout = createDescriptorSetLayout;
        result.DestroyDescriptorSetLayout = destroyDescriptorSetLayout;
        result.CreatePipelineLayout = createPipelineLayout;
        result.DestroyPipelineLayout = destroyPipelineLayout;
        result.CreatePipelineCache = createPipelineCache;
        result.DestroyPipelineCache = destroyPipelineCache;
        result.GetPipelineCacheData = getPipelineCacheData;
        result.CreateComputePipelines = createComputePipelines;
        result.DestroyPipeline = destroyPipeline;
        result.GetMemoryFdKHR = getMemoryFd;
        result.ImportSemaphoreFdKHR = importSemaphoreFd;
        result.GetSemaphoreFdKHR = getSemaphoreFd;
        return result;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL queueSubmit(VkQueue queue, uint32_t count,
            const VkSubmitInfo* submits, VkFence fence) {
        auto& self = *active;
        const auto call = observeSubmit(queue, count, submits, fence);
        if (self.fullBackendMode && !self.returnResourcesArmed) {
            self.frontSubmits.push_back(call);
            return self.submitResult;
        }
        if (queue == self.renderQueue) {
            const auto previousCalls = self.aObserved.calls;
            self.aObserved = call;
            self.aObserved.calls = previousCalls + 1;
            self.aQueueSubmitCalls = self.aObserved.calls;
            if (self.aSubmitResult == VK_SUCCESS && call.signalCount == 1
                    && call.signal == self.returnedForGraphics)
                ++self.returnedForGraphicsCalls;
            if (self.aSubmitResult == VK_SUCCESS)
                ++self.foreignReadbackPendingCalls;
            return self.aSubmitResult;
        }
        {
            auto& out = self.observed;
            const auto previousCalls = out.calls;
            out = call;
            out.calls = previousCalls + 1;
        }
        return self.submitResult;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL enumeratePhysicalDevices(
            VkInstance, uint32_t* count, VkPhysicalDevice* devices) {
        if (!devices) {
            *count = 1;
            return VK_SUCCESS;
        }
        if (*count != 0) devices[0] = active->generationPhysicalDevice;
        *count = 1;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL enumerateDeviceExtensionProperties(
            VkPhysicalDevice, const char*, uint32_t* count,
            VkExtensionProperties*) {
        *count = 0;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceProperties2(
            VkPhysicalDevice, VkPhysicalDeviceProperties2* properties) {
        properties->properties.apiVersion = VK_API_VERSION_1_2;
        properties->properties.vendorID = 0xB0B3;
        properties->properties.deviceID = 0xA3;
        properties->properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
        std::strncpy(properties->properties.deviceName, "B0B3 deterministic test device",
            VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
        for (auto* next = static_cast<VkBaseOutStructure*>(properties->pNext);
                next; next = next->pNext) {
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES) {
                auto* id = reinterpret_cast<VkPhysicalDeviceIDProperties*>(next);
                id->deviceUUID[VK_UUID_SIZE - 1] = 0xB3;
                id->driverUUID[VK_UUID_SIZE - 1] = 0xA3;
            }
        }
    }

    static VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceQueueFamilyProperties(
            VkPhysicalDevice, uint32_t* count, VkQueueFamilyProperties* properties) {
        if (!properties) {
            *count = generationQueueFamily + 1;
            return;
        }
        for (uint32_t i = 0; i < *count; ++i) properties[i] = {};
        if (*count > generationQueueFamily) {
            properties[generationQueueFamily].queueFlags =
                VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
            properties[generationQueueFamily].queueCount = 1;
        }
    }

    static VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceFeatures2(
            VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
        features->features = {};
    }

    static VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceMemoryProperties(
            VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* properties) {
        *properties = {};
        properties->memoryTypeCount = 2;
        properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        properties->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        properties->memoryHeapCount = 1;
        properties->memoryHeaps[0].size = 1ULL << 30;
        properties->memoryTypes[0].heapIndex = 0;
        properties->memoryTypes[1].heapIndex = 0;
    }

    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL getDeviceProcAddr(
            VkDevice, const char* name) {
        if (std::strcmp(name, "vkGetMemoryFdPropertiesKHR") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(getMemoryFdProperties);
        if (std::strcmp(name, "vkGetImageDrmFormatModifierPropertiesEXT") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(getImageDrmFormatModifierProperties);
        if (std::strcmp(name, "vkGetMemoryFdKHR") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(getMemoryFd);
        if (std::strcmp(name, "vkCmdCopyImage") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(cmdCopyImage);
        if (std::strcmp(name, "vkGetImageSubresourceLayout") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(getImageSubresourceLayout);
        return nullptr;
    }

    static VKAPI_ATTR void VKAPI_CALL getDeviceQueue(
            VkDevice, uint32_t family, uint32_t, VkQueue* value) {
        *value = family == generationQueueFamily ? active->queue : VK_NULL_HANDLE;
    }

    static VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceFormatProperties2(
            VkPhysicalDevice, VkFormat, VkFormatProperties2* properties) {
        auto& self = *active;
        ++self.formatPropertyQueries;
        if (self.fullBackendMode && !self.returnResourcesArmed) {
            properties->formatProperties.optimalTilingFeatures =
                VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT
                | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
                | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
                | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
            auto* next = static_cast<VkBaseOutStructure*>(properties->pNext);
            if (!next) return;
            if (next->sType
                    == VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT) {
                auto* list = reinterpret_cast<VkDrmFormatModifierPropertiesList2EXT*>(next);
                if (!list->pDrmFormatModifierProperties) {
                    list->drmFormatModifierCount = 1;
                    return;
                }
                list->drmFormatModifierCount = 1;
                list->pDrmFormatModifierProperties[0].drmFormatModifier = 0;
                list->pDrmFormatModifierProperties[0].drmFormatModifierPlaneCount = 1;
                list->pDrmFormatModifierProperties[0].drmFormatModifierTilingFeatures =
                    VK_FORMAT_FEATURE_2_BLIT_SRC_BIT
                    | VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
                return;
            }
        }
        auto* list = static_cast<VkDrmFormatModifierPropertiesListEXT*>(properties->pNext);
        if (!list) return;
        if (!list->pDrmFormatModifierProperties) {
            list->drmFormatModifierCount = 1;
            return;
        }
        list->drmFormatModifierCount = 1;
        list->pDrmFormatModifierProperties[0].drmFormatModifier = 0;
        list->pDrmFormatModifierProperties[0].drmFormatModifierPlaneCount = 1;
        list->pDrmFormatModifierProperties[0].drmFormatModifierTilingFeatures =
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL getPhysicalDeviceImageFormatProperties2(
            VkPhysicalDevice, const VkPhysicalDeviceImageFormatInfo2*,
            VkImageFormatProperties2* properties) {
        auto& self = *active;
        ++self.imageFormatPropertyQueries;
        auto* external = static_cast<VkExternalImageFormatProperties*>(properties->pNext);
        if (external) {
            external->externalMemoryProperties.externalMemoryFeatures =
                VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
                | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
            external->externalMemoryProperties.exportFromImportedHandleTypes =
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            external->externalMemoryProperties.compatibleHandleTypes =
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        }
        return VK_SUCCESS;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createImage(VkDevice device,
            const VkImageCreateInfo* info, const VkAllocationCallbacks*, VkImage* image) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed) {
            *image = self.nextHandle<VkImage>();
            return VK_SUCCESS;
        }
        ObservedImageCreate observed{};
        observed.device = device;
        observed.format = info->format;
        observed.extent = info->extent;
        observed.usage = info->usage;
        observed.tiling = info->tiling;
        for (auto* next = static_cast<const VkBaseInStructure*>(info->pNext);
                next; next = next->pNext) {
            if (next->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
                const auto* external =
                    reinterpret_cast<const VkExternalMemoryImageCreateInfo*>(next);
                observed.externalHandleTypes = external->handleTypes;
            } else if (next->sType
                    == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT) {
                const auto* modifier = reinterpret_cast<
                    const VkImageDrmFormatModifierListCreateInfoEXT*>(next);
                if (modifier->drmFormatModifierCount)
                    observed.modifier = modifier->pDrmFormatModifiers[0];
            } else if (next->sType
                    == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT) {
                const auto* modifier = reinterpret_cast<
                    const VkImageDrmFormatModifierExplicitCreateInfoEXT*>(next);
                observed.explicitModifier = true;
                observed.modifier = modifier->drmFormatModifier;
                observed.modifierPlaneCount = modifier->drmFormatModifierPlaneCount;
                if (modifier->drmFormatModifierPlaneCount)
                    observed.plane = modifier->pPlaneLayouts[0];
            }
        }

        const bool consumer = device == self.renderDevice;
        const bool importedOnA = consumer && observed.explicitModifier;
        const auto requestedFailure = consumer
            ? ShadowHarnessFailurePoint::CONSUMER_IMAGE_CREATE
            : ShadowHarnessFailurePoint::RETURN_IMAGE_CREATE;
        if (importedOnA
                && self.aFailurePoint == ShadowAHarnessFailurePoint::IMAGE_CREATE) {
            observed.result = VK_ERROR_INITIALIZATION_FAILED;
            self.imageCreates.push_back(observed);
            return observed.result;
        }
        if (self.failurePoint == requestedFailure) {
            observed.result = VK_ERROR_INITIALIZATION_FAILED;
            self.imageCreates.push_back(observed);
            return observed.result;
        }
        observed.image = importedOnA ? self.aImportedImage
            : (consumer ? self.consumerProbeImage : self.returnImage);
        *image = observed.image;
        self.imageCreates.push_back(observed);
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL destroyImage(
            VkDevice, VkImage image, const VkAllocationCallbacks*) {
        if (active->fullBackendMode && image != active->returnImage
                && image != active->consumerProbeImage
                && image != active->aImportedImage)
            return;
        active->destroyedImages.push_back(image);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL getImageDrmFormatModifierProperties(
            VkDevice, VkImage, VkImageDrmFormatModifierPropertiesEXT* properties) {
        ++active->modifierQueries;
        properties->drmFormatModifier = 0;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL getImageSubresourceLayout(
            VkDevice, VkImage, const VkImageSubresource*, VkSubresourceLayout* layout) {
        ++active->subresourceLayoutQueries;
        *layout = active->returnPlane;
    }

    static VKAPI_ATTR void VKAPI_CALL getImageMemoryRequirements2(VkDevice,
            const VkImageMemoryRequirementsInfo2* info,
            VkMemoryRequirements2* requirements) {
        auto& self = *active;
        auto& observed = info->image == self.aImportedImage
            ? self.aMemoryRequirements : self.memoryRequirements;
        ++observed.calls;
        observed.image = info->image;
        observed.size = returnAllocationSize;
        observed.alignment = returnAllocationAlignment;
        observed.memoryTypeBits = returnMemoryTypeBits;
        requirements->memoryRequirements = {
            returnAllocationSize, returnAllocationAlignment, returnMemoryTypeBits};
        auto* dedicated = static_cast<VkMemoryDedicatedRequirements*>(requirements->pNext);
        if (dedicated) {
            dedicated->prefersDedicatedAllocation = VK_FALSE;
            dedicated->requiresDedicatedAllocation = VK_FALSE;
        }
    }

    static VKAPI_ATTR void VKAPI_CALL getImageMemoryRequirements(
            VkDevice, VkImage, VkMemoryRequirements* requirements) {
        *requirements = {returnAllocationSize, returnAllocationAlignment, 0x3};
    }

    static VKAPI_ATTR VkResult VKAPI_CALL allocateMemory(VkDevice allocationDevice,
            const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
            VkDeviceMemory* memory) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed) {
            *memory = self.nextHandle<VkDeviceMemory>();
            return VK_SUCCESS;
        }
        if (allocationDevice == self.renderDevice) {
            ObservedMemoryAllocation observed{};
            observed.calls = 1;
            observed.size = info->allocationSize;
            observed.memoryTypeIndex = info->memoryTypeIndex;
            const VkImportMemoryFdInfoKHR* importInfo{};
            for (auto* next = static_cast<const VkBaseInStructure*>(info->pNext);
                    next; next = next->pNext) {
                if (next->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR)
                    importInfo = reinterpret_cast<const VkImportMemoryFdInfoKHR*>(next);
                if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO)
                    observed.dedicated = true;
            }
            if (importInfo) {
                if (self.aFailurePoint == ShadowAHarnessFailurePoint::MEMORY_IMPORT) {
                    observed.result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
                    self.aMemoryAllocations.push_back(observed);
                    return observed.result;
                }
                observed.memory = self.aImportedMemory;
                *memory = observed.memory;
                if (importInfo->fd >= 0) ::close(importInfo->fd);
                self.aExternalMemoryImported = true;
            } else if (!self.aExternalMemoryImported) {
                observed.memory = self.aPlainProbeMemory;
                *memory = observed.memory;
            } else {
                if (self.aFailurePoint
                        == ShadowAHarnessFailurePoint::STAGING_MEMORY_ALLOCATION) {
                    observed.result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
                    self.aMemoryAllocations.push_back(observed);
                    return observed.result;
                }
                observed.memory = self.aStagingMemory;
                *memory = observed.memory;
            }
            self.aMemoryAllocations.push_back(observed);
            return VK_SUCCESS;
        }
        auto& observed = self.memoryAllocation;
        ++observed.calls;
        observed.size = info->allocationSize;
        observed.memoryTypeIndex = info->memoryTypeIndex;
        const auto* exportInfo = static_cast<const VkExportMemoryAllocateInfo*>(info->pNext);
        if (exportInfo) {
            observed.exportHandleTypes = exportInfo->handleTypes;
            observed.dedicated = exportInfo->pNext != nullptr;
        }
        if (self.failurePoint == ShadowHarnessFailurePoint::MEMORY_ALLOCATION) {
            observed.result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            return observed.result;
        }
        observed.memory = self.returnMemory;
        *memory = observed.memory;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL freeMemory(
            VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
        if (memory == active->aPlainProbeMemory || memory == active->aImportedMemory
                || memory == active->aStagingMemory) {
            active->aFreedMemories.push_back(memory);
            return;
        }
        if (active->fullBackendMode && memory != active->returnMemory)
            return;
        ++active->freedMemories;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL bindImageMemory(
            VkDevice, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed)
            return VK_SUCCESS;
        if (image == self.aImportedImage) {
            ++self.aImageBind.calls;
            self.aImageBind.image = image;
            self.aImageBind.memory = memory;
            self.aImageBind.offset = offset;
            return VK_SUCCESS;
        }
        auto& observed = self.imageBind;
        ++observed.calls;
        observed.image = image;
        observed.memory = memory;
        observed.offset = offset;
        if (self.failurePoint == ShadowHarnessFailurePoint::IMAGE_BIND) {
            observed.result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
            return observed.result;
        }
        return VK_SUCCESS;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL getMemoryFd(VkDevice,
            const VkMemoryGetFdInfoKHR* info, int* fd) {
        auto& self = *active;
        ++self.memoryFdExports;
        self.lastExportedMemory = info->memory;
        self.lastMemoryExportHandleType = info->handleType;
        if (self.memoryExportResult != VK_SUCCESS) return self.memoryExportResult;
        int ends[2]{};
        if (::pipe(ends) != 0) return VK_ERROR_OUT_OF_HOST_MEMORY;
        ::close(ends[1]);
        self.lastMemoryExportFd = ends[0];
        *fd = ends[0];
        return VK_SUCCESS;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createCommandPool(VkDevice,
            const VkCommandPoolCreateInfo* info, const VkAllocationCallbacks*,
            VkCommandPool* pool) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed) {
            if (info->queueFamilyIndex != generationQueueFamily)
                return VK_ERROR_INITIALIZATION_FAILED;
            *pool = self.nextHandle<VkCommandPool>();
            return VK_SUCCESS;
        }
        const bool aSide = info->queueFamilyIndex == renderQueueFamily;
        auto& observed = aSide
            ? self.aCommandPoolObservation : self.commandPoolObservation;
        ++observed.createCalls;
        observed.queueFamilyIndex = info->queueFamilyIndex;
        observed.flags = info->flags;
        if (aSide && self.aFailurePoint
                == ShadowAHarnessFailurePoint::COMMAND_POOL_CREATE)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        if (!aSide && self.failurePoint == ShadowHarnessFailurePoint::COMMAND_POOL_CREATE)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        observed.pool = aSide ? self.aCommandPool : self.returnCommandPool;
        *pool = observed.pool;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL destroyCommandPool(
            VkDevice, VkCommandPool pool, const VkAllocationCallbacks*) {
        if (pool == active->aCommandPool) {
            ++active->aCommandPoolObservation.destroyCalls;
            return;
        }
        if (active->fullBackendMode && pool != active->returnCommandPool)
            return;
        ++active->commandPoolObservation.destroyCalls;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL allocateCommandBuffers(VkDevice,
            const VkCommandBufferAllocateInfo* info, VkCommandBuffer* commands) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed) {
            for (uint32_t i = 0; i < info->commandBufferCount; ++i)
                commands[i] = self.nextHandle<VkCommandBuffer>();
            return VK_SUCCESS;
        }
        const bool aSide = info->commandPool == self.aCommandPool;
        auto& observed = aSide ? self.aCommandAllocation : self.commandAllocation;
        ++observed.calls;
        observed.pool = info->commandPool;
        observed.level = info->level;
        observed.count = info->commandBufferCount;
        if (aSide && self.aFailurePoint
                == ShadowAHarnessFailurePoint::COMMAND_BUFFER_ALLOCATION)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        if (!aSide
                && self.failurePoint == ShadowHarnessFailurePoint::COMMAND_BUFFER_ALLOCATION)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        observed.command = aSide ? self.aCommand : self.returnCommand;
        commands[0] = observed.command;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL freeCommandBuffers(VkDevice, VkCommandPool pool,
            uint32_t count, const VkCommandBuffer* commands) {
        if (pool == active->aCommandPool) {
            auto& observed = active->aCommandAllocation;
            ++observed.freeCalls;
            observed.freedCount += count;
            observed.pool = pool;
            return;
        }
        if (active->fullBackendMode && pool != active->returnCommandPool)
            return;
        auto& observed = active->commandAllocation;
        ++observed.freeCalls;
        observed.freedCount += count;
        observed.pool = pool;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL beginCommandBuffer(
            VkCommandBuffer command, const VkCommandBufferBeginInfo* info) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed)
            return VK_SUCCESS;
        const bool aSide = command == self.aCommand;
        auto& observed = aSide ? self.aCommandBeginEnd : self.commandBeginEnd;
        ++observed.beginCalls;
        observed.beginCommand = command;
        observed.beginFlags = info->flags;
        (aSide ? self.aCommandTrace : self.commandTrace).emplace_back("BEGIN_CB");
        if (aSide && self.aFailurePoint == ShadowAHarnessFailurePoint::BEGIN_COMMAND_BUFFER)
            return VK_ERROR_INITIALIZATION_FAILED;
        return !aSide && self.failurePoint == ShadowHarnessFailurePoint::BEGIN_COMMAND_BUFFER
            ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL cmdPipelineBarrier(VkCommandBuffer command,
            VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage,
            VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount,
            const VkMemoryBarrier*, uint32_t bufferBarrierCount,
            const VkBufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
            const VkImageMemoryBarrier* imageBarriers) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed)
            return;
        ObservedPipelineBarrier observed{};
        observed.command = command;
        observed.sourceStage = sourceStage;
        observed.destinationStage = destinationStage;
        observed.dependencyFlags = dependencyFlags;
        observed.memoryBarrierCount = memoryBarrierCount;
        observed.bufferBarrierCount = bufferBarrierCount;
        observed.imageBarrierCount = imageBarrierCount;
        if (bufferBarrierCount) observed.bufferBarrier = bufferBarriers[0];
        if (imageBarrierCount) observed.imageBarrier = imageBarriers[0];
        const bool aSide = command == self.aCommand;
        (aSide ? self.aBarriers : self.barriers).push_back(observed);
        (aSide ? self.aCommandTrace : self.commandTrace).emplace_back("BARRIER");
    }

    static VKAPI_ATTR void VKAPI_CALL cmdCopyImage(VkCommandBuffer command,
            VkImage source, VkImageLayout sourceLayout, VkImage destination,
            VkImageLayout destinationLayout, uint32_t regionCount,
            const VkImageCopy* regions) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed)
            return;
        auto& observed = self.imageCopy;
        ++observed.calls;
        observed.command = command;
        observed.source = source;
        observed.sourceLayout = sourceLayout;
        observed.destination = destination;
        observed.destinationLayout = destinationLayout;
        observed.regionCount = regionCount;
        if (regionCount) observed.region = regions[0];
        self.commandTrace.emplace_back("COPY_IMAGE");
    }

    static VKAPI_ATTR VkResult VKAPI_CALL endCommandBuffer(VkCommandBuffer command) {
        auto& self = *active;
        if (self.fullBackendMode && !self.returnResourcesArmed)
            return VK_SUCCESS;
        const bool aSide = command == self.aCommand;
        auto& observed = aSide ? self.aCommandBeginEnd : self.commandBeginEnd;
        ++observed.endCalls;
        observed.endCommand = command;
        (aSide ? self.aCommandTrace : self.commandTrace).emplace_back("END_CB");
        if (aSide && self.aFailurePoint == ShadowAHarnessFailurePoint::END_COMMAND_BUFFER)
            return VK_ERROR_INITIALIZATION_FAILED;
        return !aSide && self.failurePoint == ShadowHarnessFailurePoint::END_COMMAND_BUFFER
            ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createDescriptorPool(VkDevice,
            const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*,
            VkDescriptorPool* value) {
        *value = active->nextHandle<VkDescriptorPool>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyDescriptorPool(
            VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {}

    static VKAPI_ATTR VkResult VKAPI_CALL createBuffer(VkDevice bufferDevice,
            const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer* value) {
        if (bufferDevice == active->renderDevice && active->returnResourcesArmed) {
            ++active->aBufferCreateCalls;
            if (active->aFailurePoint
                    == ShadowAHarnessFailurePoint::STAGING_BUFFER_CREATE)
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            *value = active->aStagingBuffer;
            return VK_SUCCESS;
        }
        *value = active->nextHandle<VkBuffer>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyBuffer(
            VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
        if (buffer == active->aStagingBuffer) ++active->aBufferDestroyCalls;
    }
    static VKAPI_ATTR void VKAPI_CALL getBufferMemoryRequirements(
            VkDevice, VkBuffer, VkMemoryRequirements* requirements) {
        *requirements = {returnAllocationSize, returnAllocationAlignment, 0x2};
    }
    static VKAPI_ATTR VkResult VKAPI_CALL bindBufferMemory(
            VkDevice, VkBuffer buffer, VkDeviceMemory, VkDeviceSize) {
        if (buffer == active->aStagingBuffer) ++active->aBufferBindCalls;
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL mapMemory(VkDevice mapDevice, VkDeviceMemory,
            VkDeviceSize, VkDeviceSize size, VkMemoryMapFlags, void** data) {
        if (size != VK_WHOLE_SIZE && size > active->mappedStorage.size())
            return VK_ERROR_MEMORY_MAP_FAILED;
        if (mapDevice == active->renderDevice) ++active->aMapCalls;
        *data = active->mappedStorage.data();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL unmapMemory(VkDevice mapDevice, VkDeviceMemory) {
        if (mapDevice == active->renderDevice) ++active->aUnmapCalls;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL invalidateMappedMemoryRanges(
            VkDevice, uint32_t, const VkMappedMemoryRange*) {
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL cmdFillBuffer(
            VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t) {}
    static VKAPI_ATTR void VKAPI_CALL cmdCopyBuffer(VkCommandBuffer, VkBuffer,
            VkBuffer, uint32_t, const VkBufferCopy*) {}
    static VKAPI_ATTR void VKAPI_CALL cmdBlitImage(VkCommandBuffer, VkImage,
            VkImageLayout, VkImage, VkImageLayout, uint32_t,
            const VkImageBlit*, VkFilter) {}
    static VKAPI_ATTR void VKAPI_CALL cmdClearColorImage(VkCommandBuffer, VkImage,
            VkImageLayout, const VkClearColorValue*, uint32_t,
            const VkImageSubresourceRange*) {}
    static VKAPI_ATTR void VKAPI_CALL cmdBindPipeline(
            VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
    static VKAPI_ATTR void VKAPI_CALL cmdBindDescriptorSets(VkCommandBuffer,
            VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t,
            const VkDescriptorSet*, uint32_t, const uint32_t*) {}
    static VKAPI_ATTR void VKAPI_CALL cmdDispatch(
            VkCommandBuffer, uint32_t, uint32_t, uint32_t) {}
    static VKAPI_ATTR void VKAPI_CALL cmdCopyBufferToImage(VkCommandBuffer,
            VkBuffer, VkImage, VkImageLayout, uint32_t, const VkBufferImageCopy*) {}
    static VKAPI_ATTR void VKAPI_CALL cmdCopyImageToBuffer(VkCommandBuffer command,
            VkImage source, VkImageLayout layout, VkBuffer destination,
            uint32_t regionCount, const VkBufferImageCopy* regions) {
        if (active->fullBackendMode && !active->returnResourcesArmed) return;
        auto& observed = active->aImageToBuffer;
        ++observed.calls;
        observed.command = command;
        observed.source = source;
        observed.sourceLayout = layout;
        observed.destination = destination;
        observed.regionCount = regionCount;
        if (regionCount) observed.region = regions[0];
        active->aCommandTrace.emplace_back("COPY_IMAGE_TO_BUFFER");
    }

    static VKAPI_ATTR VkResult VKAPI_CALL allocateDescriptorSets(VkDevice,
            const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
        for (uint32_t i = 0; i < info->descriptorSetCount; ++i)
            sets[i] = active->nextHandle<VkDescriptorSet>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL freeDescriptorSets(
            VkDevice, VkDescriptorPool, uint32_t, const VkDescriptorSet*) {
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL updateDescriptorSets(
            VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t,
            const VkCopyDescriptorSet*) {}

    static VKAPI_ATTR VkResult VKAPI_CALL createImageView(VkDevice,
            const VkImageViewCreateInfo*, const VkAllocationCallbacks*, VkImageView* value) {
        *value = active->nextHandle<VkImageView>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyImageView(
            VkDevice, VkImageView, const VkAllocationCallbacks*) {}
    static VKAPI_ATTR VkResult VKAPI_CALL createSampler(VkDevice,
            const VkSamplerCreateInfo*, const VkAllocationCallbacks*, VkSampler* value) {
        *value = active->nextHandle<VkSampler>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroySampler(
            VkDevice, VkSampler, const VkAllocationCallbacks*) {}
    static VKAPI_ATTR VkResult VKAPI_CALL createShaderModule(VkDevice,
            const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*,
            VkShaderModule* value) {
        *value = active->nextHandle<VkShaderModule>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyShaderModule(
            VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
    static VKAPI_ATTR VkResult VKAPI_CALL createDescriptorSetLayout(VkDevice,
            const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*,
            VkDescriptorSetLayout* value) {
        *value = active->nextHandle<VkDescriptorSetLayout>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyDescriptorSetLayout(
            VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}
    static VKAPI_ATTR VkResult VKAPI_CALL createPipelineLayout(VkDevice,
            const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*,
            VkPipelineLayout* value) {
        *value = active->nextHandle<VkPipelineLayout>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyPipelineLayout(
            VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
    static VKAPI_ATTR VkResult VKAPI_CALL createPipelineCache(VkDevice,
            const VkPipelineCacheCreateInfo*, const VkAllocationCallbacks*,
            VkPipelineCache* value) {
        *value = active->nextHandle<VkPipelineCache>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyPipelineCache(
            VkDevice, VkPipelineCache, const VkAllocationCallbacks*) {}
    static VKAPI_ATTR VkResult VKAPI_CALL getPipelineCacheData(
            VkDevice, VkPipelineCache, size_t* size, void* data) {
        if (!data) *size = 0;
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL createComputePipelines(VkDevice,
            VkPipelineCache, uint32_t count, const VkComputePipelineCreateInfo*,
            const VkAllocationCallbacks*, VkPipeline* pipelines) {
        for (uint32_t i = 0; i < count; ++i)
            pipelines[i] = active->nextHandle<VkPipeline>();
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyPipeline(
            VkDevice, VkPipeline, const VkAllocationCallbacks*) {}

    static VKAPI_ATTR VkResult VKAPI_CALL getMemoryFdProperties(VkDevice queryDevice,
            VkExternalMemoryHandleTypeFlagBits, int, VkMemoryFdPropertiesKHR* properties) {
        if (queryDevice == active->renderDevice) ++active->aMemoryFdPropertyCalls;
        properties->memoryTypeBits = 1;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createSemaphore(VkDevice semaphoreDevice,
            const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* value) {
        if (active->fullBackendMode && !active->returnResourcesArmed) {
            *value = active->nextHandle<VkSemaphore>();
            return VK_SUCCESS;
        }
        if (semaphoreDevice == active->renderDevice) {
            ++active->aSemaphoreCreateCalls;
            if (active->aFailurePoint
                    == ShadowAHarnessFailurePoint::SEMAPHORE_CREATE)
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            *value = active->aImportedWait;
            return VK_SUCCESS;
        }
        ++active->semaphoreCreateCalls;
        *value = active->exportSignal;
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroySemaphore(
            VkDevice semaphoreDevice, VkSemaphore semaphore, const VkAllocationCallbacks*) {
        if (semaphoreDevice == active->renderDevice || semaphore == active->aImportedWait) {
            ++active->aSemaphoreDestroyCalls;
            return;
        }
        if (active->fullBackendMode && semaphore != active->exportSignal)
            return;
        ++active->semaphoreDestroyCalls;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL getSemaphoreFd(VkDevice,
            const VkSemaphoreGetFdInfoKHR* info, int* fd) {
        auto& self = *active;
        ++self.exportCalls;
        if (self.returnResourcesArmed) ++self.productionBReturnExportCalls;
        self.lastExportedSemaphore = info->semaphore;
        if (self.exportResult != VK_SUCCESS) return self.exportResult;
        int ends[2]{};
        if (::pipe(ends) != 0) return VK_ERROR_OUT_OF_HOST_MEMORY;
        ::close(ends[1]);
        *fd = ends[0];
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL importSemaphoreFd(VkDevice importDevice,
            const VkImportSemaphoreFdInfoKHR* info) {
        if (importDevice == active->renderDevice) {
            auto& observed = active->aSemaphoreImport;
            ++observed.calls;
            observed.device = importDevice;
            observed.semaphore = info->semaphore;
            observed.flags = info->flags;
            observed.handleType = info->handleType;
            observed.fd = info->fd;
            if (active->aFailurePoint == ShadowAHarnessFailurePoint::SEMAPHORE_IMPORT)
                observed.result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
            else
                observed.result = active->aImportResult;
            ++active->aImportCalls;
            if (observed.result == VK_SUCCESS && info->fd >= 0) ::close(info->fd);
            return observed.result;
        }
        ++active->backendImportedFdCalls;
        if (info->fd >= 0) ::close(info->fd);
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL createFence(VkDevice fenceDevice,
            const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* value) {
        if (active->fullBackendMode && !active->returnResourcesArmed) {
            *value = active->nextHandle<VkFence>();
            return VK_SUCCESS;
        }
        if (fenceDevice == active->renderDevice) {
            ++active->aFenceCreateCalls;
            if (active->aFailurePoint == ShadowAHarnessFailurePoint::FENCE_CREATE)
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            *value = active->aReturnFence;
            return VK_SUCCESS;
        }
        ++active->fenceCreateCalls;
        *value = active->returnFence;
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyFence(
            VkDevice fenceDevice, VkFence fence, const VkAllocationCallbacks*) {
        if (fenceDevice == active->renderDevice || fence == active->aReturnFence) {
            ++active->aFenceDestroyCalls;
            return;
        }
        if (active->fullBackendMode && fence != active->returnFence)
            return;
        ++active->fenceDestroyCalls;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL waitForFences(VkDevice, uint32_t count,
            const VkFence* fences, VkBool32, uint64_t) {
        if (active->fullBackendMode && !active->returnResourcesArmed) {
            ++active->backendFenceWaitCalls;
            return VK_SUCCESS;
        }
        const VkFence fence = count && fences ? fences[0] : VK_NULL_HANDLE;
        if (active->stagedProduction
                && fence == active->stagedProduction->sourceReadRetirementFence) {
            ++active->sourceFenceWaitCalls;
            return active->sourceFenceWaitResult;
        }
        if (fence == active->aReturnFence) {
            ++active->aFenceWaitCalls;
            return active->aFenceWaitResult;
        }
        ++active->fenceWaitCalls;
        return active->fenceWaitResult;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL resetFences(
            VkDevice, uint32_t, const VkFence*) { return VK_SUCCESS; }
    static VKAPI_ATTR VkResult VKAPI_CALL deviceWaitIdle(VkDevice) {
        ++active->deviceIdleCalls;
        return VK_SUCCESS;
    }
};

}
