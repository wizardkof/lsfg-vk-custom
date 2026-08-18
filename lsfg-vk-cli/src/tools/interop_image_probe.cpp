/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "interop_image_probe.hpp"
#include "lsfg-vk-common/vulkan/external_capabilities.hpp"
#include "lsfg-vk-common/vulkan/external_buffer_transport.hpp"
#include "lsfg-vk-common/vulkan/external_semaphore_sync.hpp"
#include "lsfg-vk-common/vulkan/external_image_transport.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"

#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <gbm.h>
#include <sys/stat.h>
#include <unistd.h>
#include <memory>
#include <vulkan/vulkan_core.h>

namespace probe = lsfgvk::cli::interop_image_probe;
namespace {
constexpr VkImageUsageFlags USAGE = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkExternalMemoryHandleTypeFlagBits HANDLE =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
struct Candidate { VkFormat format; const char* name; };

std::vector<vk::ImageModifierInfo> enumerateModifiers(
        const vk::VulkanInstanceInventoryFuncs& funcs, VkPhysicalDevice device,
        VkFormat format) {
    uint32_t count{};
    VkDrmFormatModifierPropertiesList2EXT list{
        .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
        .drmFormatModifierCount = 0,
        .pDrmFormatModifierProperties = nullptr
    };
    VkFormatProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        .pNext = &list
    };
    funcs.GetPhysicalDeviceFormatProperties2(device, format, &properties);
    count = list.drmFormatModifierCount;
    std::vector<VkDrmFormatModifierProperties2EXT> values(count);
    list.pDrmFormatModifierProperties = values.data();
    funcs.GetPhysicalDeviceFormatProperties2(device, format, &properties);
    std::vector<vk::ImageModifierInfo> result;
    for (const auto& value : values)
        result.push_back({value.drmFormatModifier, value.drmFormatModifierPlaneCount,
            value.drmFormatModifierTilingFeatures});
    return result;
}

vk::ExternalImageCapability queryModifierImage(
        const vk::VulkanInstanceInventoryFuncs& funcs, VkPhysicalDevice device,
        VkFormat format, uint64_t modifier, VkImageUsageFlags usage, VkExtent3D extent) {
    VkExternalImageFormatProperties external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES
    };
    VkPhysicalDeviceExternalImageFormatInfo externalInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
        .handleType = HANDLE
    };
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
        .pNext = &externalInfo,
        .drmFormatModifier = modifier,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    VkPhysicalDeviceImageFormatInfo2 info{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext = &modifierInfo,
        .format = format,
        .type = VK_IMAGE_TYPE_2D,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = usage,
        .flags = 0
    };
    VkImageFormatProperties2 output{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = &external
    };
    const auto result = funcs.GetPhysicalDeviceImageFormatProperties2(device, &info, &output);
    return {
        .query = {{format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
            usage, 0, extent, 1, 1, VK_SAMPLE_COUNT_1_BIT}, HANDLE},
        .result = result,
        .limits = output.imageFormatProperties,
        .externalMemoryFeatures = external.externalMemoryProperties.externalMemoryFeatures,
        .exportFromImportedHandleTypes = external.externalMemoryProperties.exportFromImportedHandleTypes,
        .compatibleHandleTypes = external.externalMemoryProperties.compatibleHandleTypes
    };
}
}

std::optional<probe::Options> probe::parse(const std::vector<std::string>& args,
        std::string& error) {
    Options out; bool allocator = false, a = false, b = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i + 1 >= args.size()) { error = "missing option value"; return std::nullopt; }
        try {
            if (args[i] == "--allocator") { out.allocator = args[++i]; allocator = true; }
            else if (args[i] == "--device-a") { out.deviceA = std::stoul(args[++i]); a = true; }
            else if (args[i] == "--device-b") { out.deviceB = std::stoul(args[++i]); b = true; }
            else { error = "unknown option: " + args[i]; return std::nullopt; }
        } catch (...) { error = "invalid option value"; return std::nullopt; }
    }
    if (!allocator || !a || !b) {
        error = "--allocator, --device-a and --device-b are required";
        return std::nullopt;
    }
    if (out.deviceA == out.deviceB) { error = "device A and B must differ"; return std::nullopt; }
    return out;
}

int probe::run(const Options& options) {
    try {
        vk::VulkanInventoryInstance instance{"lsfg-vk-cli", {2, 0, 0},
            "lsfg-vk-cli", {2, 0, 0}};
        const auto physical = vk::enumeratePhysicalDevices(instance.fi(), instance.inst());
        if (options.deviceA >= physical.size() || options.deviceB >= physical.size()) {
            std::cerr << "FINAL: DEVICE_INDEX_OUT_OF_RANGE\n"; return EXIT_FAILURE;
        }
        const auto identityA = vk::getPhysicalDeviceIdentity(instance.fi(), physical[options.deviceA]);
        const auto identityB = vk::getPhysicalDeviceIdentity(instance.fi(), physical[options.deviceB]);
        std::cout << "[DG2X-P4A] Cross-device VkImage transport\n"
            << "Device A: " << identityA.name << "\n"
            << "Device B: " << identityB.name << "\n"
            << "Allocator node: " << options.allocator << "\n";

        const Candidate formats[] = {
            {VK_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM"},
            {VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM"},
            {VK_FORMAT_R8_UNORM, "R8_UNORM"},
            {VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT"}
        };
        std::vector<vk::ImageTransportContract> contracts;
        const VkExtent3D extent{256, 256, 1};
        const auto extensionA = vk::enumerateDeviceExtensionNames(instance.fi(), physical[options.deviceA]);
        const auto extensionB = vk::enumerateDeviceExtensionNames(instance.fi(), physical[options.deviceB]);
        const bool modifiersAvailable = std::ranges::find(extensionA,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) != extensionA.end()
            && std::ranges::find(extensionB,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) != extensionB.end();
        if (!modifiersAvailable) {
            std::cout << "DRM modifier extension A/B: NO\n"
                << "FINAL: BLOCKED_BY_DRM_MODIFIER_CONTRACT\n";
            return EXIT_SUCCESS;
        }
        for (const auto& format : formats) {
            const auto modifiersA = enumerateModifiers(instance.fi(), physical[options.deviceA], format.format);
            const auto modifiersB = enumerateModifiers(instance.fi(), physical[options.deviceB], format.format);
            std::cout << "FORMAT " << format.name << "\n"
                << "AMD modifier count: " << modifiersA.size() << "\n";
            for (const auto& m : modifiersA)
                std::cout << "  AMD modifier=0x" << std::hex << m.modifier << std::dec
                    << " planes=" << m.planeCount << " features=0x" << std::hex
                    << m.features << std::dec << "\n";
            std::cout << "NVIDIA modifier count: " << modifiersB.size() << "\n";
            for (const auto& m : modifiersB)
                std::cout << "  NVIDIA modifier=0x" << std::hex << m.modifier << std::dec
                    << " planes=" << m.planeCount << " features=0x" << std::hex
                    << m.features << std::dec << "\n";
            const auto common = vk::filterImageModifiers(
                vk::intersectImageModifiers(modifiersA, modifiersB),
                VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT);
            std::cout << "Common modifiers: " << common.size() << "\n";
            const auto selectedModifier = vk::selectImageModifier(common);
            if (selectedModifier) {
                const auto a = queryModifierImage(instance.fi(), physical[options.deviceA],
                    format.format, selectedModifier->modifier, USAGE, extent);
                const auto b = queryModifierImage(instance.fi(), physical[options.deviceB],
                    format.format, selectedModifier->modifier, USAGE, extent);
                contracts.push_back({
                    .candidate = {format.format, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                        USAGE, extent, HANDLE, selectedModifier->modifier,
                        selectedModifier->planeCount},
                    .a = {a.result, a.externalMemoryFeatures,
                        a.compatibleHandleTypes, a.limits},
                    .b = {b.result, b.externalMemoryFeatures,
                        b.compatibleHandleTypes, b.limits}
                });
                std::cout << "  Candidate query modifier=0x" << std::hex
                    << selectedModifier->modifier << std::dec
                    << " A=" << (a.result == VK_SUCCESS ? "PASS" : "FAIL")
                    << " B=" << (b.result == VK_SUCCESS ? "PASS" : "FAIL") << "\n";
            }
        }
        const auto selected = vk::chooseImageTransportContract(contracts);
        if (!selected) {
            std::cout << "FINAL: BLOCKED_BY_DRM_MODIFIER_CONTRACT\n";
            return EXIT_SUCCESS;
        }
        const auto fourcc = vk::drmFourccForVkFormat(selected->candidate.format);
        if (!fourcc) {
            std::cout << "FINAL: FAIL_CONTRACT_FORMAT_MAPPING\n";
            return EXIT_FAILURE;
        }
        const int node = ::open(options.allocator.c_str(), O_RDWR | O_CLOEXEC);
        if (node < 0) {
            std::cout << "GBM allocation: BLOCKED errno=" << errno << "\n"
                << "FINAL: BLOCKED_BY_ALLOCATOR\n";
            return EXIT_SUCCESS;
        }
        gbm_device* gbm = gbm_create_device(node);
        const uint64_t modifier = selected->candidate.modifier;
        const uint64_t modifiers[] = {modifier};
        gbm_bo* bo = gbm ? gbm_bo_create_with_modifiers2(gbm, 256, 256, *fourcc,
            modifiers, 1, GBM_BO_USE_RENDERING) : nullptr;
        if (!bo) {
            std::cout << "GBM allocation: BLOCKED errno=" << errno << "\n"
                << "FINAL: BLOCKED_BY_ALLOCATOR\n";
            if (gbm) gbm_device_destroy(gbm);
            ::close(node);
            return EXIT_SUCCESS;
        }
        const auto actualFourcc = gbm_bo_get_format(bo);
        const auto actualModifier = gbm_bo_get_modifier(bo);
        const auto actualPlanes = gbm_bo_get_plane_count(bo);
        std::vector<VkSubresourceLayout> layouts;
        for (uint32_t plane = 0; plane < actualPlanes; ++plane) {
            layouts.push_back({
                .offset = gbm_bo_get_offset(bo, static_cast<int>(plane)),
                .size = 0,
                .rowPitch = gbm_bo_get_stride_for_plane(bo, static_cast<int>(plane)),
                .arrayPitch = 0,
                .depthPitch = 0
            });
        }
        const bool metadata = vk::validateImagePlaneMetadata(*fourcc, modifier, 1,
            actualFourcc, actualModifier, actualPlanes, layouts);
        std::cout << "Selected format: " << static_cast<int>(selected->candidate.format) << "\n"
            << "DRM fourcc: 0x" << std::hex << *fourcc << std::dec << "\n"
            << "DRM modifier: 0x" << std::hex << actualModifier << std::dec << "\n"
            << "Plane count: " << actualPlanes << "\n"
            << "Stride: " << (layouts.empty() ? 0 : layouts.front().rowPitch) << "\n"
            << "Offset: " << (layouts.empty() ? 0 : layouts.front().offset) << "\n"
            << "GBM allocation: PASS\n"
            << "GBM modifier match: " << (metadata ? "PASS" : "FAIL") << "\n"
            << "Plane metadata validation: " << (metadata ? "PASS" : "FAIL") << "\n";
        if (!metadata) {
            gbm_bo_destroy(bo); gbm_device_destroy(gbm); ::close(node);
            std::cout << "FINAL: BLOCKED_BY_ALLOCATOR_CONTRACT_MISMATCH\n";
            return EXIT_SUCCESS;
        }

        const auto selectA = [index = options.deviceA](const auto&, const auto& devices) {
            return devices.at(index);
        };
        const auto selectB = [index = options.deviceB](const auto&, const auto& devices) {
            return devices.at(index);
        };
        vk::Vulkan deviceA{"lsfg-vk-cli", {2, 0, 0}, "lsfg-vk-cli", {2, 0, 0}, selectA};
        vk::Vulkan deviceB{"lsfg-vk-cli", {2, 0, 0}, "lsfg-vk-cli", {2, 0, 0}, selectB};
        auto createAndImport = [&](vk::Vulkan& device, int fd, const char* label) {
            const auto& capability = label[0] == 'A' ? selected->a : selected->b;
            const bool externalDedicated = (capability.features
                & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
            std::cout << label << " external IMPORTABLE: "
                << ((capability.features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) ? "YES" : "NO")
                << " EXPORTABLE: "
                << ((capability.features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) ? "YES" : "NO")
                << " DEDICATED_ONLY: " << (externalDedicated ? "YES" : "NO") << "\n";
            const VkExternalMemoryImageCreateInfo external{
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .handleTypes = HANDLE
            };
            const VkImageDrmFormatModifierExplicitCreateInfoEXT explicitModifier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                .pNext = &external,
                .drmFormatModifier = modifier,
                .drmFormatModifierPlaneCount = static_cast<uint32_t>(actualPlanes),
                .pPlaneLayouts = layouts.data()
            };
            const VkImageCreateInfo imageInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = &explicitModifier,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = selected->candidate.format,
                .extent = extent,
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                .usage = USAGE,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
            };
            VkImage image{};
            auto result = device.df().CreateImage(device.dev(), &imageInfo, nullptr, &image);
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string(label) + "_IMAGE_CREATE_FAILED VkResult="
                    + std::to_string(result));
            if (!device.df().GetImageMemoryRequirements2)
                throw std::runtime_error(std::string(label) + "_IMAGE_REQUIREMENTS2_FUNCTION_MISSING");
            VkMemoryDedicatedRequirements dedicated{
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS
            };
            VkMemoryRequirements2 requirements2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                .pNext = &dedicated
            };
            const VkImageMemoryRequirementsInfo2 requirementsInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                .image = image
            };
            device.df().GetImageMemoryRequirements2(device.dev(), &requirementsInfo, &requirements2);
            const auto& requirements = requirements2.memoryRequirements;
            const bool mustDedicated = vk::imageNeedsDedicatedAllocation(
                capability.features, dedicated);
            std::cout << label << " prefers dedicated: "
                << (dedicated.prefersDedicatedAllocation ? "YES" : "NO")
                << " requires dedicated: "
                << (dedicated.requiresDedicatedAllocation ? "YES" : "NO")
                << " allocation mode: " << (mustDedicated ? "DEDICATED" : "NON_DEDICATED") << "\n";
            auto getFdProps = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
                device.fi().GetDeviceProcAddr(device.dev(), "vkGetMemoryFdPropertiesKHR"));
            if (!getFdProps)
                throw std::runtime_error(std::string(label) + "_FD_PROPERTIES_FUNCTION_MISSING");
            VkMemoryFdPropertiesKHR fdProperties{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
            result = getFdProps(device.dev(), HANDLE, fd, &fdProperties);
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string(label) + "_FD_PROPERTIES_FAILED VkResult="
                    + std::to_string(result));
            const uint32_t intersection = requirements.memoryTypeBits & fdProperties.memoryTypeBits;
            const auto memoryType = vk::selectExternalBufferMemoryType(
                intersection, [&device]() { VkPhysicalDeviceMemoryProperties p{};
                    device.fi().GetPhysicalDeviceMemoryProperties(device.physdev(), &p); return p; }(),
                0, 0);
            std::cout << label << " requirements size=" << requirements.size
                << " alignment=" << requirements.alignment << " typeBits=0x" << std::hex
                << requirements.memoryTypeBits << " fdBits=0x" << fdProperties.memoryTypeBits
                << " intersection=0x" << intersection << std::dec << "\n";
            if (!memoryType)
                throw std::runtime_error(std::string(label) + "_IMAGE_MEMORY_TYPE_INTERSECTION_FAILED");
            const VkImportMemoryFdInfoKHR importInfo{
                .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                .handleType = HANDLE,
                .fd = fd
            };
            VkMemoryDedicatedAllocateInfo dedicatedAllocate{
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                .pNext = nullptr,
                .image = image,
                .buffer = VK_NULL_HANDLE
            };
            VkMemoryAllocateInfo allocate{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = mustDedicated ? static_cast<const void*>(&dedicatedAllocate)
                    : static_cast<const void*>(&importInfo),
                .allocationSize = requirements.size,
                .memoryTypeIndex = *memoryType
            };
            if (mustDedicated)
                dedicatedAllocate.pNext = &importInfo;
            VkDeviceMemory memory{};
            result = device.df().AllocateMemory(device.dev(), &allocate, nullptr, &memory);
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string(label) + "_IMAGE_IMPORT_FAILED VkResult="
                    + std::to_string(result));
            result = device.df().BindImageMemory(device.dev(), image, memory, 0);
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string(label) + "_IMAGE_BIND_FAILED VkResult="
                    + std::to_string(result));
            std::cout << label << " memoryType=" << *memoryType << " image create/import/bind PASS\n";
            return std::pair<VkImage, VkDeviceMemory>{image, memory};
        };
        const int fdA = gbm_bo_get_fd_for_plane(bo, 0);
        const int fdB = gbm_bo_get_fd_for_plane(bo, 0);
        if (fdA < 0 || fdB < 0)
            throw std::runtime_error("GBM_IMAGE_PLANE_FD_FAILED");
        const auto imageA = createAndImport(deviceA, fdA, "A");
        const auto imageB = createAndImport(deviceB, fdB, "B");

        struct Staging {
            VkDevice device{}; VkBuffer buffer{}; VkDeviceMemory memory{};
            VkMemoryPropertyFlags flags{}; const vk::Vulkan* vk{};
        };
        auto makeStaging = [](vk::Vulkan& device) {
            Staging result{.device = device.dev(), .vk = &device};
            const VkBufferCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = 256 * 256 * 4,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
            };
            auto r = device.df().CreateBuffer(device.dev(), &info, nullptr, &result.buffer);
            if (r != VK_SUCCESS) throw std::runtime_error("STAGING_CREATE_FAILED");
            VkMemoryRequirements req{};
            device.df().GetBufferMemoryRequirements(device.dev(), result.buffer, &req);
            VkPhysicalDeviceMemoryProperties props{};
            device.fi().GetPhysicalDeviceMemoryProperties(device.physdev(), &props);
            const auto type = vk::selectExternalBufferMemoryType(req.memoryTypeBits, props,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!type) throw std::runtime_error("STAGING_MEMORY_TYPE_FAILED");
            result.flags = props.memoryTypes[*type].propertyFlags;
            const VkMemoryAllocateInfo alloc{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = req.size, .memoryTypeIndex = *type};
            r = device.df().AllocateMemory(device.dev(), &alloc, nullptr, &result.memory);
            if (r != VK_SUCCESS) throw std::runtime_error("STAGING_ALLOC_FAILED");
            r = device.df().BindBufferMemory(device.dev(), result.buffer, result.memory, 0);
            if (r != VK_SUCCESS) throw std::runtime_error("STAGING_BIND_FAILED");
            return result;
        };
        auto stagingA = makeStaging(deviceA);
        auto stagingB = makeStaging(deviceB);
        auto semaphoreDevice = [](vk::Vulkan& device) {
            return vk::ExternalSemaphoreDevice{.device = device.dev(), .funcs = {
                .CreateSemaphore = device.df().CreateSemaphore,
                .DestroySemaphore = device.df().DestroySemaphore,
                .GetSemaphoreFdKHR = device.df().GetSemaphoreFdKHR,
                .ImportSemaphoreFdKHR = device.df().ImportSemaphoreFdKHR}};
        };
        auto semA = semaphoreDevice(deviceA);
        auto semB = semaphoreDevice(deviceB);
        auto exportA = vk::createExportableSyncFdSemaphore(semA);
        auto importB = vk::createSyncFdImportSemaphore(semB);
        auto exportB = vk::createExportableSyncFdSemaphore(semB);
        auto importA = vk::createSyncFdImportSemaphore(semA);
        auto command = [](vk::Vulkan& device) {
            VkCommandPool pool{}; const VkCommandPoolCreateInfo poolInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = device.queueFamilyIndex()};
            auto r = device.df().CreateCommandPool(device.dev(), &poolInfo, nullptr, &pool);
            if (r != VK_SUCCESS) throw std::runtime_error("IMAGE_COMMAND_POOL_FAILED");
            VkCommandBuffer cmd{}; const VkCommandBufferAllocateInfo alloc{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1};
            r = device.df().AllocateCommandBuffers(device.dev(), &alloc, &cmd);
            if (r != VK_SUCCESS) throw std::runtime_error("IMAGE_COMMAND_BUFFER_FAILED");
            const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            r = device.df().BeginCommandBuffer(cmd, &begin);
            if (r != VK_SUCCESS) throw std::runtime_error("IMAGE_COMMAND_BEGIN_FAILED");
            return std::pair<VkCommandPool, VkCommandBuffer>{pool, cmd};
        };
        auto barrier = [](vk::Vulkan& device, VkCommandBuffer cmd, VkImage image,
                uint32_t srcFamily, uint32_t dstFamily, VkAccessFlags srcAccess,
                VkAccessFlags dstAccess, VkImageLayout oldLayout, VkImageLayout newLayout) {
            const VkImageMemoryBarrier b{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = srcAccess, .dstAccessMask = dstAccess,
                .oldLayout = oldLayout, .newLayout = newLayout,
                .srcQueueFamilyIndex = srcFamily, .dstQueueFamilyIndex = dstFamily,
                .image = image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            device.df().CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        const VkImageSubresourceRange colorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto submit = [](vk::Vulkan& device, VkCommandBuffer cmd, VkSemaphore wait,
                VkSemaphore signal, VkFence fence = VK_NULL_HANDLE) {
            constexpr VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = wait ? 1U : 0U, .pWaitSemaphores = wait ? &wait : nullptr,
                .pWaitDstStageMask = wait ? &stage : nullptr, .commandBufferCount = 1,
                .pCommandBuffers = &cmd, .signalSemaphoreCount = signal ? 1U : 0U,
                .pSignalSemaphores = signal ? &signal : nullptr};
            const auto r = device.df().QueueSubmit(device.queue(), 1, &info, fence);
            if (r != VK_SUCCESS) throw std::runtime_error("IMAGE_SUBMIT_FAILED");
        };
        auto renderCmd = command(deviceA);
        barrier(deviceA, renderCmd.second, imageA.first, VK_QUEUE_FAMILY_FOREIGN_EXT,
            deviceA.queueFamilyIndex(), 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const VkClearColorValue patternA{{0.6470588F, 0.6470588F, 0.6470588F, 0.6470588F}};
        deviceA.df().CmdClearColorImage(renderCmd.second, imageA.first,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &patternA, 1,
            &colorRange);
        barrier(deviceA, renderCmd.second, imageA.first, deviceA.queueFamilyIndex(),
            VK_QUEUE_FAMILY_FOREIGN_EXT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        if (deviceA.df().EndCommandBuffer(renderCmd.second) != VK_SUCCESS)
            throw std::runtime_error("IMAGE_COMMAND_END_FAILED");
        submit(deviceA, renderCmd.second, VK_NULL_HANDLE, exportA.handle());
        auto payloadAB = vk::exportSyncFd(semA, exportA.handle());
        vk::importSyncFdTemporary(semB, importB.handle(), payloadAB);

        auto generationCmd = command(deviceB);
        barrier(deviceB, generationCmd.second, imageB.first, VK_QUEUE_FAMILY_FOREIGN_EXT,
            deviceB.queueFamilyIndex(), 0, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        const VkBufferImageCopy copyB{0, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {256, 256, 1}};
        deviceB.df().CmdCopyImageToBuffer(generationCmd.second, imageB.first,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingB.buffer, 1, &copyB);
        barrier(deviceB, generationCmd.second, imageB.first, deviceB.queueFamilyIndex(),
            deviceB.queueFamilyIndex(), VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const VkClearColorValue patternB{{0.3529412F, 0.3529412F, 0.3529412F, 0.3529412F}};
        deviceB.df().CmdClearColorImage(generationCmd.second, imageB.first,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &patternB, 1,
            &colorRange);
        barrier(deviceB, generationCmd.second, imageB.first, deviceB.queueFamilyIndex(),
            VK_QUEUE_FAMILY_FOREIGN_EXT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        if (deviceB.df().EndCommandBuffer(generationCmd.second) != VK_SUCCESS)
            throw std::runtime_error("IMAGE_COMMAND_END_FAILED");
        submit(deviceB, generationCmd.second, importB.handle(), exportB.handle());
        auto payloadBA = vk::exportSyncFd(semB, exportB.handle());
        vk::importSyncFdTemporary(semA, importA.handle(), payloadBA);

        auto finalCmd = command(deviceA);
        barrier(deviceA, finalCmd.second, imageA.first, VK_QUEUE_FAMILY_FOREIGN_EXT,
            deviceA.queueFamilyIndex(), 0, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        const VkBufferImageCopy copyA = copyB;
        deviceA.df().CmdCopyImageToBuffer(finalCmd.second, imageA.first,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingA.buffer, 1, &copyA);
        if (deviceA.df().EndCommandBuffer(finalCmd.second) != VK_SUCCESS)
            throw std::runtime_error("IMAGE_COMMAND_END_FAILED");
        VkFence fence{}; const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (deviceA.df().CreateFence(deviceA.dev(), &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("IMAGE_FENCE_FAILED");
        submit(deviceA, finalCmd.second, importA.handle(), VK_NULL_HANDLE, fence);
        if (deviceA.df().WaitForFences(deviceA.dev(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            throw std::runtime_error("IMAGE_FINAL_WAIT_FAILED");
        deviceA.df().DestroyFence(deviceA.dev(), fence, nullptr);
        auto verify = [](vk::Vulkan& device, const Staging& staging, uint32_t expected) {
            void* mapped{};
            auto r = device.df().MapMemory(device.dev(), staging.memory, 0, VK_WHOLE_SIZE, 0, &mapped);
            if (r != VK_SUCCESS) throw std::runtime_error("IMAGE_STAGING_MAP_FAILED");
            if ((staging.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                    nullptr, staging.memory, 0, VK_WHOLE_SIZE};
                r = device.df().InvalidateMappedMemoryRanges(device.dev(), 1, &range);
                if (r != VK_SUCCESS) throw std::runtime_error("IMAGE_STAGING_INVALIDATE_FAILED");
            }
            const auto* words = static_cast<const uint32_t*>(mapped);
            for (size_t i = 0; i < (256U * 256U); ++i)
                if (words[i] != expected) {
                    device.df().UnmapMemory(device.dev(), staging.memory);
                    return false;
                }
            device.df().UnmapMemory(device.dev(), staging.memory);
            return true;
        };
        const bool observedA = verify(deviceB, stagingB, 0xA5A5A5A5);
        const bool observedB = verify(deviceA, stagingA, 0x5A5A5A5A);
        if (!observedA || !observedB)
            throw std::runtime_error("IMAGE_CONTENT_MISMATCH");
        std::cout << "A image pattern written: PASS\nB observed A image: PASS\n"
            << "B image pattern written: PASS\nA observed B image: PASS\n"
            << "SYNC_FD A->B: PASS\nSYNC_FD B->A: PASS\n"
            << "Host waits before final submit: NONE\nSafe teardown: PASS\n"
            << "FINAL: CROSS_DEVICE_DMA_BUF_IMAGE_PASS\n";
        deviceA.df().FreeMemory(deviceA.dev(), imageA.second, nullptr);
        deviceA.df().DestroyImage(deviceA.dev(), imageA.first, nullptr);
        deviceB.df().FreeMemory(deviceB.dev(), imageB.second, nullptr);
        deviceB.df().DestroyImage(deviceB.dev(), imageB.first, nullptr);
        ::close(fdA); ::close(fdB);
        gbm_bo_destroy(bo); gbm_device_destroy(gbm); ::close(node);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "FINAL: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
