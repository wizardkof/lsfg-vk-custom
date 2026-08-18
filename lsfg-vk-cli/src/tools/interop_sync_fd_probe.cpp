/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "interop_sync_fd_probe.hpp"

#include "lsfg-vk-common/helpers/owned_fd.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <gbm.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vulkan/vulkan_core.h>

namespace probe = lsfgvk::cli::interop_sync_fd_probe;

namespace {
constexpr VkDeviceSize SIZE = 64 * 1024;
constexpr uint32_t PATTERN_A = 0xA5A5A5A5;
constexpr uint32_t PATTERN_B = 0x5A5A5A5A;
constexpr uint64_t FINAL_FENCE_TIMEOUT_NS = 10'000'000'000ULL;
constexpr const char* SEM_FD = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;

struct ProbeFailure : std::runtime_error {
    ProbeFailure(probe::Failure failure, std::string detail = {})
        : std::runtime_error(detail.empty() ? probe::failureName(failure)
                                            : std::string(probe::failureName(failure)) + ": " + detail),
          failure(failure) {}

    probe::Failure failure;
};

template<class T>
T load(PFN_vkGetDeviceProcAddr getProcAddr, VkDevice device, const char* name) {
    auto proc = reinterpret_cast<T>(getProcAddr(device, name));
    if (!proc)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string("DEVICE_FUNCTION_MISSING ") + name);
    return proc;
}

bool hasExtension(const std::vector<std::string>& names, const char* wanted) {
    return std::ranges::binary_search(names, std::string(wanted));
}

bool syncFdCapable(const vk::VulkanInstanceInventoryFuncs& funcs,
                   VkPhysicalDevice physical,
                   VkExternalSemaphoreFeatureFlags required) {
    VkPhysicalDeviceExternalSemaphoreInfo info{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO
    };
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    VkExternalSemaphoreProperties properties{
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES
    };
    funcs.GetPhysicalDeviceExternalSemaphoreProperties(physical, &info, &properties);

    return (properties.externalSemaphoreFeatures & required) == required
        && (properties.compatibleHandleTypes
            & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) != 0;
}

struct Device {
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};

    PFN_vkDestroyDevice destroy{};
    PFN_vkCreateBuffer createBuffer{};
    PFN_vkDestroyBuffer destroyBuffer{};
    PFN_vkGetBufferMemoryRequirements requirements{};
    PFN_vkAllocateMemory allocate{};
    PFN_vkFreeMemory freeMemory{};
    PFN_vkBindBufferMemory bind{};
    PFN_vkGetMemoryFdPropertiesKHR fdProperties{};
    PFN_vkCreateCommandPool createPool{};
    PFN_vkDestroyCommandPool destroyPool{};
    PFN_vkAllocateCommandBuffers allocateCommands{};
    PFN_vkBeginCommandBuffer begin{};
    PFN_vkEndCommandBuffer end{};
    PFN_vkCmdPipelineBarrier barrier{};
    PFN_vkCmdFillBuffer fill{};
    PFN_vkCmdCopyBuffer copy{};
    PFN_vkCreateFence createFence{};
    PFN_vkDestroyFence destroyFence{};
    PFN_vkWaitForFences waitFence{};
    PFN_vkQueueSubmit submit{};
    PFN_vkMapMemory map{};
    PFN_vkUnmapMemory unmap{};
    PFN_vkInvalidateMappedMemoryRanges invalidate{};
    PFN_vkCreateSemaphore createSemaphore{};
    PFN_vkDestroySemaphore destroySemaphore{};
    PFN_vkGetSemaphoreFdKHR getSemaphoreFd{};
    PFN_vkImportSemaphoreFdKHR importSemaphoreFd{};

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    ~Device() {
        if (device && destroy)
            destroy(device, nullptr);
    }
};

struct Buffer {
    Device* device{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    VkMemoryPropertyFlags flags{};

    explicit Buffer(Device* device = nullptr) : device(device) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept
        : device(std::exchange(other.device, nullptr)),
          buffer(std::exchange(other.buffer, VK_NULL_HANDLE)),
          memory(std::exchange(other.memory, VK_NULL_HANDLE)),
          flags(other.flags) {}

    ~Buffer() {
        if (device && buffer)
            device->destroyBuffer(device->device, buffer, nullptr);
        if (device && memory)
            device->freeMemory(device->device, memory, nullptr);
    }
};

struct CommandWork {
    Device* device{};
    VkCommandPool pool{};
    VkCommandBuffer command{};

    explicit CommandWork(Device* device = nullptr) : device(device) {}
    CommandWork(const CommandWork&) = delete;
    CommandWork& operator=(const CommandWork&) = delete;
    CommandWork(CommandWork&& other) noexcept
        : device(std::exchange(other.device, nullptr)),
          pool(std::exchange(other.pool, VK_NULL_HANDLE)),
          command(std::exchange(other.command, VK_NULL_HANDLE)) {}

    ~CommandWork() {
        if (device && pool)
            device->destroyPool(device->device, pool, nullptr);
    }
};

struct Semaphore {
    Device* device{};
    VkSemaphore semaphore{};

    explicit Semaphore(Device* device = nullptr) : device(device) {}
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore(Semaphore&& other) noexcept
        : device(std::exchange(other.device, nullptr)),
          semaphore(std::exchange(other.semaphore, VK_NULL_HANDLE)) {}

    ~Semaphore() {
        if (device && semaphore)
            device->destroySemaphore(device->device, semaphore, nullptr);
    }
};

struct Fence {
    Device* device{};
    VkFence fence{};

    explicit Fence(Device* device = nullptr) : device(device) {}
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;
    Fence(Fence&& other) noexcept
        : device(std::exchange(other.device, nullptr)),
          fence(std::exchange(other.fence, VK_NULL_HANDLE)) {}

    ~Fence() {
        if (device && fence)
            device->destroyFence(device->device, fence, nullptr);
    }
};

struct GbmPayload {
    ls::OwnedFd node;
    gbm_device* device{};
    gbm_bo* bo{};
    ls::OwnedFd fd;
    VkDeviceSize size{};

    ~GbmPayload() {
        fd.reset();
        if (bo)
            gbm_bo_destroy(bo);
        if (device)
            gbm_device_destroy(device);
    }
};

struct SyncFdPayload {
    bool valid{};
    bool sentinel{};
    ls::OwnedFd fd;

    [[nodiscard]] int nativeFd() const noexcept {
        return sentinel ? -1 : fd.get();
    }
};

struct PatternResult {
    bool matches{};
    size_t mismatchIndex{};
};

std::unique_ptr<Device> makeDevice(const vk::VulkanInstanceInventoryFuncs& funcs,
                                   VkPhysicalDevice physical,
                                   const char* label) {
    const auto extensions = vk::enumerateDeviceExtensionNames(funcs, physical);
    const char* requiredExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };
    for (const char* extension : requiredExtensions) {
        if (!hasExtension(extensions, extension))
            throw ProbeFailure(probe::Failure::ExtensionMissing,
                std::string(label) + " missing " + extension);
    }

    uint32_t queueCount{};
    funcs.GetPhysicalDeviceQueueFamilyProperties(physical, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queueCount);
    funcs.GetPhysicalDeviceQueueFamilyProperties(physical, &queueCount, queues.data());
    const auto queueIt = std::ranges::find_if(queues, [](const auto& properties) {
        return (properties.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
    });
    if (queueIt == queues.end())
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " has no transfer queue");

    const uint32_t family = static_cast<uint32_t>(queueIt - queues.begin());
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = family;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = std::size(requiredExtensions);
    createInfo.ppEnabledExtensionNames = requiredExtensions;

    auto owner = std::make_unique<Device>();
    Device& device = *owner;
    device.physical = physical;
    device.family = family;

    const VkResult result = funcs.CreateDevice(physical, &createInfo, nullptr, &device.device);
    if (result != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " VkDevice create VkResult=" + std::to_string(result));

    const auto getProcAddr = funcs.GetDeviceProcAddr;
    device.destroy = load<PFN_vkDestroyDevice>(getProcAddr, device.device, "vkDestroyDevice");
    device.createBuffer = load<PFN_vkCreateBuffer>(getProcAddr, device.device, "vkCreateBuffer");
    device.destroyBuffer = load<PFN_vkDestroyBuffer>(getProcAddr, device.device, "vkDestroyBuffer");
    device.requirements = load<PFN_vkGetBufferMemoryRequirements>(getProcAddr, device.device, "vkGetBufferMemoryRequirements");
    device.allocate = load<PFN_vkAllocateMemory>(getProcAddr, device.device, "vkAllocateMemory");
    device.freeMemory = load<PFN_vkFreeMemory>(getProcAddr, device.device, "vkFreeMemory");
    device.bind = load<PFN_vkBindBufferMemory>(getProcAddr, device.device, "vkBindBufferMemory");
    device.fdProperties = load<PFN_vkGetMemoryFdPropertiesKHR>(getProcAddr, device.device, "vkGetMemoryFdPropertiesKHR");
    device.createPool = load<PFN_vkCreateCommandPool>(getProcAddr, device.device, "vkCreateCommandPool");
    device.destroyPool = load<PFN_vkDestroyCommandPool>(getProcAddr, device.device, "vkDestroyCommandPool");
    device.allocateCommands = load<PFN_vkAllocateCommandBuffers>(getProcAddr, device.device, "vkAllocateCommandBuffers");
    device.begin = load<PFN_vkBeginCommandBuffer>(getProcAddr, device.device, "vkBeginCommandBuffer");
    device.end = load<PFN_vkEndCommandBuffer>(getProcAddr, device.device, "vkEndCommandBuffer");
    device.barrier = load<PFN_vkCmdPipelineBarrier>(getProcAddr, device.device, "vkCmdPipelineBarrier");
    device.fill = load<PFN_vkCmdFillBuffer>(getProcAddr, device.device, "vkCmdFillBuffer");
    device.copy = load<PFN_vkCmdCopyBuffer>(getProcAddr, device.device, "vkCmdCopyBuffer");
    device.createFence = load<PFN_vkCreateFence>(getProcAddr, device.device, "vkCreateFence");
    device.destroyFence = load<PFN_vkDestroyFence>(getProcAddr, device.device, "vkDestroyFence");
    device.waitFence = load<PFN_vkWaitForFences>(getProcAddr, device.device, "vkWaitForFences");
    device.submit = load<PFN_vkQueueSubmit>(getProcAddr, device.device, "vkQueueSubmit");
    device.map = load<PFN_vkMapMemory>(getProcAddr, device.device, "vkMapMemory");
    device.unmap = load<PFN_vkUnmapMemory>(getProcAddr, device.device, "vkUnmapMemory");
    device.invalidate = load<PFN_vkInvalidateMappedMemoryRanges>(getProcAddr, device.device, "vkInvalidateMappedMemoryRanges");
    device.createSemaphore = load<PFN_vkCreateSemaphore>(getProcAddr, device.device, "vkCreateSemaphore");
    device.destroySemaphore = load<PFN_vkDestroySemaphore>(getProcAddr, device.device, "vkDestroySemaphore");
    device.getSemaphoreFd = load<PFN_vkGetSemaphoreFdKHR>(getProcAddr, device.device, "vkGetSemaphoreFdKHR");
    device.importSemaphoreFd = load<PFN_vkImportSemaphoreFdKHR>(getProcAddr, device.device, "vkImportSemaphoreFdKHR");
    const auto getQueue = load<PFN_vkGetDeviceQueue>(getProcAddr, device.device, "vkGetDeviceQueue");
    getQueue(device.device, family, 0, &device.queue);
    funcs.GetPhysicalDeviceMemoryProperties(physical, &device.memoryProperties);
    return owner;
}

VkBuffer createRawBuffer(Device& device, bool external, VkDeviceSize size) {
    VkExternalMemoryBufferCreateInfo externalInfo{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO
    };
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkBufferCreateInfo createInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.pNext = external ? &externalInfo : nullptr;
    createInfo.size = size;
    createInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer{};
    const VkResult result = device.createBuffer(device.device, &createInfo, nullptr, &buffer);
    if (result != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            "BUFFER_CREATE_FAILED VkResult=" + std::to_string(result));
    return buffer;
}

std::optional<uint32_t> selectMemoryType(Device& device,
                                         uint32_t bits,
                                         VkMemoryPropertyFlags required,
                                         VkMemoryPropertyFlags preferred = 0) {
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < device.memoryProperties.memoryTypeCount; ++i) {
            const auto flags = device.memoryProperties.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) == 0 || (flags & required) != required)
                continue;
            if (pass == 0 && (flags & preferred) != preferred)
                continue;
            return i;
        }
    }
    return std::nullopt;
}

Buffer importSharedBuffer(Device& device,
                          ls::OwnedFd& fd,
                          VkDeviceSize backingSize,
                          const char* label) {
    Buffer out{&device};
    out.buffer = createRawBuffer(device, true, SIZE);

    VkMemoryRequirements requirements{};
    device.requirements(device.device, out.buffer, &requirements);

    VkMemoryFdPropertiesKHR fdProperties{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    const VkResult fdResult = device.fdProperties(device.device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd.get(), &fdProperties);
    std::cout << label << " requirements size=" << requirements.size
              << " alignment=" << requirements.alignment
              << " typeBits=0x" << std::hex << requirements.memoryTypeBits
              << " fdBits=0x" << fdProperties.memoryTypeBits << std::dec << "\n";
    if (fdResult != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " FD properties VkResult=" + std::to_string(fdResult));
    if (backingSize < requirements.size)
        throw ProbeFailure(probe::Failure::UnexpectedState, "DMA_BUF_SIZE_INVALID");

    const uint32_t usableBits = fdProperties.memoryTypeBits & requirements.memoryTypeBits;
    const auto memoryType = selectMemoryType(device, usableBits, 0);
    if (!memoryType)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " NO_MEMORY_TYPE_INTERSECTION");

    out.flags = device.memoryProperties.memoryTypes[*memoryType].propertyFlags;
    VkImportMemoryFdInfoKHR importInfo{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    importInfo.fd = fd.get();

    VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocationInfo.pNext = &importInfo;
    allocationInfo.allocationSize = backingSize;
    allocationInfo.memoryTypeIndex = *memoryType;

    const VkResult allocateResult = device.allocate(device.device,
        &allocationInfo, nullptr, &out.memory);
    if (allocateResult != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " DMA_BUF import VkResult=" + std::to_string(allocateResult));

    static_cast<void>(fd.release());
    const VkResult bindResult = device.bind(device.device, out.buffer, out.memory, 0);
    if (bindResult != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " buffer bind VkResult=" + std::to_string(bindResult));

    std::cout << label << " intersection=0x" << std::hex << usableBits << std::dec
              << " memoryType=" << *memoryType
              << " flags=0x" << std::hex << out.flags << std::dec << "\n";
    return out;
}

Buffer createStaging(Device& device) {
    Buffer out{&device};
    out.buffer = createRawBuffer(device, false, SIZE);

    VkMemoryRequirements requirements{};
    device.requirements(device.device, out.buffer, &requirements);
    const auto memoryType = selectMemoryType(device, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!memoryType)
        throw ProbeFailure(probe::Failure::UnexpectedState, "STAGING_NO_MEMORY_TYPE");

    out.flags = device.memoryProperties.memoryTypes[*memoryType].propertyFlags;
    VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocationInfo.allocationSize = requirements.size;
    allocationInfo.memoryTypeIndex = *memoryType;
    if (device.allocate(device.device, &allocationInfo, nullptr, &out.memory) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState, "STAGING_ALLOC_FAILED");
    if (device.bind(device.device, out.buffer, out.memory, 0) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState, "STAGING_BIND_FAILED");
    return out;
}

CommandWork beginWork(Device& device, const char* label) {
    CommandWork work{&device};
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = device.family;
    if (device.createPool(device.device, &poolInfo, nullptr, &work.pool) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " COMMAND_POOL_FAILED");

    VkCommandBufferAllocateInfo allocationInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
    };
    allocationInfo.commandPool = work.pool;
    allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocationInfo.commandBufferCount = 1;
    if (device.allocateCommands(device.device, &allocationInfo, &work.command) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " COMMAND_BUFFER_ALLOC_FAILED");

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (device.begin(work.command, &beginInfo) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState,
            std::string(label) + " COMMAND_BUFFER_BEGIN_FAILED");
    return work;
}

void acquireForeign(Device& device,
                    VkCommandBuffer command,
                    VkBuffer shared,
                    VkAccessFlags dstAccess) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
    barrier.dstQueueFamilyIndex = device.family;
    barrier.buffer = shared;
    barrier.offset = 0;
    barrier.size = SIZE;
    device.barrier(command,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void releaseForeign(Device& device,
                    VkCommandBuffer command,
                    VkBuffer shared,
                    VkAccessFlags srcAccess) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = device.family;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
    barrier.buffer = shared;
    barrier.offset = 0;
    barrier.size = SIZE;
    device.barrier(command,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void copyBuffer(Device& device, VkCommandBuffer command, VkBuffer src, VkBuffer dst) {
    const VkBufferCopy copy{0, 0, SIZE};
    device.copy(command, src, dst, 1, &copy);
}

void transferReadToWrite(Device& device, VkCommandBuffer command) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    device.barrier(command,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

CommandWork recordAFirst(Device& device, VkBuffer shared) {
    auto work = beginWork(device, "A_FIRST");
    acquireForeign(device, work.command, shared, VK_ACCESS_TRANSFER_WRITE_BIT);
    device.fill(work.command, shared, 0, SIZE, PATTERN_A);
    releaseForeign(device, work.command, shared, VK_ACCESS_TRANSFER_WRITE_BIT);
    if (device.end(work.command) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState, "A_FIRST COMMAND_BUFFER_END_FAILED");
    return work;
}

CommandWork recordB(Device& device, VkBuffer shared, VkBuffer staging) {
    auto work = beginWork(device, "B");
    acquireForeign(device, work.command, shared, VK_ACCESS_TRANSFER_READ_BIT);
    copyBuffer(device, work.command, shared, staging);
    transferReadToWrite(device, work.command);
    device.fill(work.command, shared, 0, SIZE, PATTERN_B);
    releaseForeign(device, work.command, shared, VK_ACCESS_TRANSFER_WRITE_BIT);
    if (device.end(work.command) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState, "B COMMAND_BUFFER_END_FAILED");
    return work;
}

CommandWork recordAFinal(Device& device, VkBuffer shared, VkBuffer staging) {
    auto work = beginWork(device, "A_FINAL");
    acquireForeign(device, work.command, shared, VK_ACCESS_TRANSFER_READ_BIT);
    copyBuffer(device, work.command, shared, staging);
    if (device.end(work.command) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState, "A_FINAL COMMAND_BUFFER_END_FAILED");
    return work;
}

Semaphore createSemaphore(Device& device, bool exportable, probe::Failure failure) {
    Semaphore out{&device};
    VkExportSemaphoreCreateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    createInfo.pNext = exportable ? &exportInfo : nullptr;
    const VkResult result = device.createSemaphore(device.device, &createInfo, nullptr, &out.semaphore);
    if (result != VK_SUCCESS)
        throw ProbeFailure(failure, "VkResult=" + std::to_string(result));
    return out;
}

SyncFdPayload exportSyncFd(Device& device,
                           VkSemaphore semaphore,
                           probe::Failure failure) {
    VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    info.semaphore = semaphore;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    int nativeFd = -2;
    const VkResult result = device.getSemaphoreFd(device.device, &info, &nativeFd);
    if (result != VK_SUCCESS)
        throw ProbeFailure(failure, "VkResult=" + std::to_string(result));
    if (nativeFd < -1)
        throw ProbeFailure(failure, "driver returned invalid fd");

    SyncFdPayload payload;
    payload.valid = true;
    payload.sentinel = nativeFd == -1;
    if (nativeFd >= 0)
        payload.fd.reset(nativeFd);
    return payload;
}

void importSyncFd(Device& device,
                  VkSemaphore semaphore,
                  SyncFdPayload& payload,
                  probe::Failure failure) {
    const probe::ExportedFd publicView{payload.nativeFd(), payload.valid};
    if (!probe::acceptsSyncFd(publicView))
        throw ProbeFailure(failure, "invalid SYNC_FD payload state");

    VkImportSemaphoreFdInfoKHR info{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
    info.semaphore = semaphore;
    info.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    info.fd = payload.nativeFd();

    const VkResult result = device.importSemaphoreFd(device.device, &info);
    if (result != VK_SUCCESS)
        throw ProbeFailure(failure, "VkResult=" + std::to_string(result));

    // A successful fd import transfers ownership to Vulkan. -1 is the valid
    // already-signaled SYNC_FD sentinel and has no userspace descriptor to close.
    if (!payload.sentinel)
        static_cast<void>(payload.fd.release());
    payload.valid = false;
}

void submit(Device& device,
            VkCommandBuffer command,
            VkSemaphore waitSemaphore,
            VkSemaphore signalSemaphore,
            VkFence fence,
            probe::Failure failure) {
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (waitSemaphore) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &waitSemaphore;
        submitInfo.pWaitDstStageMask = &waitStage;
    }
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command;
    if (signalSemaphore) {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &signalSemaphore;
    }

    const VkResult result = device.submit(device.queue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS)
        throw ProbeFailure(failure, "VkResult=" + std::to_string(result));
}

Fence createFence(Device& device) {
    Fence out{&device};
    VkFenceCreateInfo createInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (device.createFence(device.device, &createInfo, nullptr, &out.fence) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::SubmitFinal, "final fence create failed");
    return out;
}

PatternResult readPattern(Device& device, Buffer& staging, uint32_t expected) {
    void* mapped{};
    if (device.map(device.device, staging.memory, 0, SIZE, 0, &mapped) != VK_SUCCESS)
        throw ProbeFailure(probe::Failure::UnexpectedState, "STAGING_MAP_FAILED");

    if ((staging.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = staging.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        if (device.invalidate(device.device, 1, &range) != VK_SUCCESS) {
            device.unmap(device.device, staging.memory);
            throw ProbeFailure(probe::Failure::UnexpectedState, "STAGING_INVALIDATE_FAILED");
        }
    }

    const auto* words = static_cast<const uint32_t*>(mapped);
    PatternResult result{true, SIZE / sizeof(uint32_t)};
    for (size_t i = 0; i < SIZE / sizeof(uint32_t); ++i) {
        if (words[i] != expected) {
            result = {false, i};
            break;
        }
    }
    device.unmap(device.device, staging.memory);
    return result;
}

} // namespace

bool probe::temporaryImport() noexcept {
    return true;
}

bool probe::acceptsSyncFd(const ExportedFd& fd) noexcept {
    return fd.valid && fd.fd >= -1;
}

const char* probe::failureName(Failure failure) noexcept {
    switch (failure) {
    case Failure::None: return "NONE";
    case Failure::Capability: return "SYNC_FD_CAPABILITY_FAILED";
    case Failure::ExtensionMissing: return "EXTERNAL_SEMAPHORE_FD_EXTENSION_MISSING";
    case Failure::ExportCreate: return "EXPORT_SEMAPHORE_CREATE_FAILED";
    case Failure::ImportCreate: return "IMPORT_SEMAPHORE_CREATE_FAILED";
    case Failure::SubmitA: return "A_SUBMIT_FAILED";
    case Failure::ExportAB: return "A_SYNC_FD_EXPORT_FAILED";
    case Failure::ImportB: return "A_SYNC_FD_IMPORT_ON_B_FAILED";
    case Failure::SubmitB: return "B_SUBMIT_FAILED";
    case Failure::ExportBA: return "B_SYNC_FD_EXPORT_FAILED";
    case Failure::ImportA: return "B_SYNC_FD_IMPORT_ON_A_FAILED";
    case Failure::SubmitFinal: return "A_FINAL_SUBMIT_FAILED";
    case Failure::DataAB: return "A_TO_B_SYNC_FD_DATA_MISMATCH";
    case Failure::DataBA: return "B_TO_A_SYNC_FD_DATA_MISMATCH";
    case Failure::UnexpectedState: return "SYNC_FD_UNEXPECTED_STATE";
    }
    return "SYNC_FD_UNEXPECTED_STATE";
}

bool probe::hasCrossDeviceHostWait(const std::vector<std::string>& events) noexcept {
    const auto finalSubmit = std::ranges::find(events, "A_FINAL_SUBMIT");
    const auto limit = finalSubmit == events.end() ? events.end() : finalSubmit;
    return std::ranges::any_of(events.begin(), limit, [](const std::string& event) {
        return event == "WAIT_A_BEFORE_B_SUBMIT"
            || event == "WAIT_B_BEFORE_A_SUBMIT"
            || event.starts_with("HOST_WAIT");
    });
}

std::optional<probe::Options> probe::parse(const std::vector<std::string>& args,
                                           std::string& error) {
    Options options;
    bool allocator = false;
    bool deviceA = false;
    bool deviceB = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i + 1 >= args.size()) {
            error = "missing option value";
            return std::nullopt;
        }
        try {
            if (args[i] == "--allocator") {
                options.allocator = args[++i];
                allocator = true;
            } else if (args[i] == "--device-a") {
                options.deviceA = std::stoul(args[++i]);
                deviceA = true;
            } else if (args[i] == "--device-b") {
                options.deviceB = std::stoul(args[++i]);
                deviceB = true;
            } else {
                error = "unknown option: " + args[i];
                return std::nullopt;
            }
        } catch (...) {
            error = "invalid option value";
            return std::nullopt;
        }
    }
    if (!allocator || !deviceA || !deviceB) {
        error = "--allocator, --device-a and --device-b are required";
        return std::nullopt;
    }
    if (options.deviceA == options.deviceB) {
        error = "device A and B must differ";
        return std::nullopt;
    }
    return options;
}

int probe::run(const Options& options) {
    try {
        GbmPayload gbm;
        gbm.node.reset(::open(options.allocator.c_str(), O_RDWR | O_CLOEXEC));
        if (!gbm.node)
            throw ProbeFailure(Failure::UnexpectedState,
                "ALLOCATOR_OPEN_FAILED errno=" + std::to_string(errno));
        gbm.device = gbm_create_device(gbm.node.get());
        if (!gbm.device)
            throw ProbeFailure(Failure::UnexpectedState,
                "GBM_DEVICE_CREATE_FAILED errno=" + std::to_string(errno));
        gbm.bo = gbm_bo_create(gbm.device, 256, 256, GBM_FORMAT_R8,
            GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING);
        if (!gbm.bo)
            throw ProbeFailure(Failure::UnexpectedState,
                "GBM_ALLOC_FAILED errno=" + std::to_string(errno));
        gbm.fd.reset(gbm_bo_get_fd(gbm.bo));
        if (!gbm.fd)
            throw ProbeFailure(Failure::UnexpectedState,
                "DMA_BUF_FD_FAILED errno=" + std::to_string(errno));

        const off_t end = lseek(gbm.fd.get(), 0, SEEK_END);
        if (end < 0)
            throw ProbeFailure(Failure::UnexpectedState,
                "DMA_BUF_SIZE_INVALID errno=" + std::to_string(errno));
        gbm.size = static_cast<VkDeviceSize>(end);
        static_cast<void>(lseek(gbm.fd.get(), 0, SEEK_SET));

        ls::OwnedFd fdA(fcntl(gbm.fd.get(), F_DUPFD_CLOEXEC, 0));
        ls::OwnedFd fdB(fcntl(gbm.fd.get(), F_DUPFD_CLOEXEC, 0));
        if (!fdA || !fdB)
            throw ProbeFailure(Failure::UnexpectedState,
                "DMA_BUF_FD_DUP_FAILED errno=" + std::to_string(errno));

        vk::VulkanInventoryInstance instance{
            "lsfg-vk-cli", {2, 0, 0}, "lsfg-vk-cli", {2, 0, 0}
        };
        const auto physical = vk::enumeratePhysicalDevices(instance.fi(), instance.inst());
        if (options.deviceA >= physical.size() || options.deviceB >= physical.size()) {
            std::cerr << "FINAL: DEVICE_INDEX_OUT_OF_RANGE\n";
            return EXIT_FAILURE;
        }

        const auto idA = vk::getPhysicalDeviceIdentity(instance.fi(), physical[options.deviceA]);
        const auto idB = vk::getPhysicalDeviceIdentity(instance.fi(), physical[options.deviceB]);
        const auto extensionsA = vk::enumerateDeviceExtensionNames(instance.fi(), physical[options.deviceA]);
        const auto extensionsB = vk::enumerateDeviceExtensionNames(instance.fi(), physical[options.deviceB]);
        const bool extA = hasExtension(extensionsA, SEM_FD);
        const bool extB = hasExtension(extensionsB, SEM_FD);
        const bool aExport = syncFdCapable(instance.fi(), physical[options.deviceA],
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT);
        const bool aImport = syncFdCapable(instance.fi(), physical[options.deviceA],
            VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT);
        const bool bExport = syncFdCapable(instance.fi(), physical[options.deviceB],
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT);
        const bool bImport = syncFdCapable(instance.fi(), physical[options.deviceB],
            VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT);

        std::cout << "Allocator: GBM\n"
                  << "Allocator node: " << options.allocator << "\n"
                  << "Logical size: " << SIZE << "\n"
                  << "DMA_BUF size: " << gbm.size << "\n"
                  << "Device A: " << idA.name << "\n"
                  << "Device B: " << idB.name << "\n"
                  << "A exportable: " << (aExport ? "YES" : "NO") << "\n"
                  << "A importable: " << (aImport ? "YES" : "NO") << "\n"
                  << "B exportable: " << (bExport ? "YES" : "NO") << "\n"
                  << "B importable: " << (bImport ? "YES" : "NO") << "\n"
                  << "A extension " << SEM_FD << ": " << (extA ? "YES" : "NO") << "\n"
                  << "B extension " << SEM_FD << ": " << (extB ? "YES" : "NO") << "\n"
                  << "Temporary import flags: VK_SEMAPHORE_IMPORT_TEMPORARY_BIT\n"
                  << "Cross-device host waits before final submit: NONE\n";

        if (!extA || !extB)
            throw ProbeFailure(Failure::ExtensionMissing);
        if (!aExport || !bImport)
            throw ProbeFailure(Failure::Capability, "A_TO_B");
        if (!bExport || !aImport)
            throw ProbeFailure(Failure::Capability, "B_TO_A");

        auto ownerA = makeDevice(instance.fi(), physical[options.deviceA], "A");
        auto ownerB = makeDevice(instance.fi(), physical[options.deviceB], "B");
        Device& a = *ownerA;
        Device& b = *ownerB;

        Buffer sharedA = importSharedBuffer(a, fdA, gbm.size, "A");
        Buffer sharedB = importSharedBuffer(b, fdB, gbm.size, "B");
        Buffer stagingA = createStaging(a);
        Buffer stagingB = createStaging(b);

        CommandWork workA = recordAFirst(a, sharedA.buffer);
        CommandWork workB = recordB(b, sharedB.buffer, stagingB.buffer);
        CommandWork workAFinal = recordAFinal(a, sharedA.buffer, stagingA.buffer);

        Semaphore signalA = createSemaphore(a, true, Failure::ExportCreate);
        Semaphore waitB = createSemaphore(b, false, Failure::ImportCreate);
        Semaphore signalB = createSemaphore(b, true, Failure::ExportCreate);
        Semaphore waitA = createSemaphore(a, false, Failure::ImportCreate);
        Fence finalFence = createFence(a);

        std::vector<std::string> events;
        submit(a, workA.command, VK_NULL_HANDLE, signalA.semaphore,
            VK_NULL_HANDLE, Failure::SubmitA);
        events.emplace_back("A_SUBMIT");

        auto syncAB = exportSyncFd(a, signalA.semaphore, Failure::ExportAB);
        std::cout << "A_TO_B_SYNC_FD_EXPORT "
                  << (syncAB.sentinel ? "SENTINEL_-1" : "FD") << "\n";
        events.emplace_back(syncAB.sentinel ? "EXPORT_AB_SENTINEL" : "EXPORT_AB_FD");
        importSyncFd(b, waitB.semaphore, syncAB, Failure::ImportB);
        events.emplace_back("IMPORT_B_TEMPORARY");

        submit(b, workB.command, waitB.semaphore, signalB.semaphore,
            VK_NULL_HANDLE, Failure::SubmitB);
        events.emplace_back("B_SUBMIT");

        auto syncBA = exportSyncFd(b, signalB.semaphore, Failure::ExportBA);
        std::cout << "B_TO_A_SYNC_FD_EXPORT "
                  << (syncBA.sentinel ? "SENTINEL_-1" : "FD") << "\n";
        events.emplace_back(syncBA.sentinel ? "EXPORT_BA_SENTINEL" : "EXPORT_BA_FD");
        importSyncFd(a, waitA.semaphore, syncBA, Failure::ImportA);
        events.emplace_back("IMPORT_A_TEMPORARY");

        submit(a, workAFinal.command, waitA.semaphore, VK_NULL_HANDLE,
            finalFence.fence, Failure::SubmitFinal);
        events.emplace_back("A_FINAL_SUBMIT");

        if (hasCrossDeviceHostWait(events))
            throw ProbeFailure(Failure::UnexpectedState, "forbidden pre-final host wait recorded");
        std::cout << "SYNC_FD_CHAIN_SUBMITTED A_TO_B_TO_A\n";

        // This is deliberately the first host wait in the A -> B -> A chain.
        // Completion of this fence implies A-first completed, B consumed A's
        // SYNC_FD and completed, and A-final consumed B's SYNC_FD and completed.
        const VkResult waitResult = a.waitFence(a.device, 1, &finalFence.fence,
            VK_TRUE, FINAL_FENCE_TIMEOUT_NS);
        if (waitResult != VK_SUCCESS)
            throw ProbeFailure(Failure::SubmitFinal,
                "final fence wait VkResult=" + std::to_string(waitResult));
        events.emplace_back("HOST_WAIT_FINAL_READBACK");

        const auto ab = readPattern(b, stagingB, PATTERN_A);
        if (!ab.matches)
            throw ProbeFailure(Failure::DataAB,
                "index=" + std::to_string(ab.mismatchIndex));
        const auto ba = readPattern(a, stagingA, PATTERN_B);
        if (!ba.matches)
            throw ProbeFailure(Failure::DataBA,
                "index=" + std::to_string(ba.mismatchIndex));

        std::cout << "A_TO_B_SYNC_FD_DATA_CHECK PASS\n"
                  << "B_TO_A_SYNC_FD_DATA_CHECK PASS\n"
                  << "FINAL: CROSS_DEVICE_SYNC_FD_PASS\n";
        return EXIT_SUCCESS;
    } catch (const ProbeFailure& failure) {
        std::cerr << "FINAL: " << failure.what() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "FINAL: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
