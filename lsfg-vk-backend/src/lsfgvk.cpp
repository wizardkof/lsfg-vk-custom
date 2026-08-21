/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-backend/lsfgvk.hpp"
#include "extraction/dll_reader.hpp"
#include "extraction/shader_registry.hpp"
#include "helpers/limits.hpp"
#include "helpers/utils.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/buffer.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/exchange_image_sync.hpp"
#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/timeline_semaphore.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "shaderchains/alpha0.hpp"
#include "shaderchains/alpha1.hpp"
#include "shaderchains/beta0.hpp"
#include "shaderchains/beta1.hpp"
#include "shaderchains/delta0.hpp"
#include "shaderchains/delta1.hpp"
#include "shaderchains/gamma0.hpp"
#include "shaderchains/gamma1.hpp"
#include "shaderchains/generate.hpp"
#include "shaderchains/mipmaps.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#ifdef LSFGVK_TESTING_RENDERDOC
#include <renderdoc_app.h>
#include <dlfcn.h>
#endif

using namespace lsfgvk;
using namespace lsfgvk::backend;

namespace lsfgvk::backend {
    error::error(const std::string& msg, const std::exception& inner)
        : std::runtime_error(msg + "\n- " + inner.what()) {}
    error::error(const std::string& msg)
        : std::runtime_error(msg) {}
    error::~error() = default;

    /// instance class
    class InstanceImpl {
    public:
        /// create an instance
        /// (see lsfg-vk documentation)
        InstanceImpl(vk::PhysicalDeviceSelector selectPhysicalDevice,
            std::shared_ptr<std::vector<vk::PhysicalDeviceSnapshot>> observedDevices,
            const std::filesystem::path& shaderDllPath,
            bool allowLowPrecision);

        /// get the Vulkan instance
        /// @return the Vulkan instance
        [[nodiscard]] const auto& getVulkan() const { return this->vk; }
        /// get the selected physical device identity
        [[nodiscard]] const auto& getDeviceIdentity() const { return this->deviceIdentity; }
        /// get all physical devices visible to the backend instance
        [[nodiscard]] const auto& getVisibleDevices() const { return this->visibleDevices; }
        /// get the shader registry
        /// @return the shader registry
        [[nodiscard]] const auto& getShaderRegistry() const { return this->shaders; }
#ifdef LSFGVK_TESTING_RENDERDOC
        /// get the RenderDoc API
        /// @return the RenderDoc API
        [[nodiscard]] const auto& getRenderDocAPI() const { return this->renderdoc; }
#endif
        // Movable, non-copyable, custom destructor
        InstanceImpl(const InstanceImpl&) = delete;
        InstanceImpl& operator=(const InstanceImpl&) = delete;
        InstanceImpl(InstanceImpl&&) = default;
        InstanceImpl& operator=(InstanceImpl&&) = default;
        ~InstanceImpl();
    private:
        vk::Vulkan vk;
        std::vector<vk::PhysicalDeviceSnapshot> visibleDevices;
        vk::PhysicalDeviceIdentity deviceIdentity;
        ShaderRegistry shaders;

#ifdef LSFGVK_TESTING_RENDERDOC
        std::optional<RENDERDOC_API_1_6_0> renderdoc;
#endif
    };

    /// context class
    class ContextImpl {
    public:
        /// create a context
        /// (see lsfg-vk documentation)
        ContextImpl(const InstanceImpl& instance,
            std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
            std::vector<vk::ExternalImage> destImages, int syncFd,
            VkExtent2D extent, bool hdr, float flow, bool perf);

        /// schedule frames
        /// (see lsfg-vk documentation)
        void scheduleFrames();
        /// schedule zero or more frames at explicit interpolation timestamps
        void scheduleFrames(const std::vector<float>& timestamps);
    private:
        void schedulePreparedFrames(size_t generatedFrames);

        std::pair<vk::Image, vk::Image> sourceImages;
        std::vector<vk::Image> destImages;
        vk::Image blackImage;

        vk::TimelineSemaphore syncSemaphore; // imported
        vk::TimelineSemaphore prepassSemaphore;
        size_t idx{1};
        size_t fidx{0}; // real frame index
        std::vector<bool> destinationFirstUse;

        std::vector<vk::CommandBuffer> cmdbufs;
        vk::Fence cmdbufFence;

        Ctx ctx;

        Mipmaps mipmaps;
        std::array<Alpha0, 7> alpha0;
        std::array<Alpha1, 7> alpha1;
        Beta0 beta0;
        Beta1 beta1;
        struct Pass {
            std::vector<Gamma0> gamma0;
            std::vector<Gamma1> gamma1;

            std::vector<Delta0> delta0;
            std::vector<Delta1> delta1;
            ls::lazy<Generate> generate;
        };
        std::vector<Pass> passes;
    };
}

Instance::Instance(
        const DevicePicker& devicePicker,
        const std::filesystem::path& shaderDllPath,
        bool allowLowPrecision,
        const DeviceEnumerationObserver& deviceObserver) {
    auto observedDevices = std::make_shared<std::vector<vk::PhysicalDeviceSnapshot>>();
    const auto selectFunc = [&devicePicker, &deviceObserver, observedDevices](
            const vk::VulkanInstanceFuncs& funcs,
            const std::vector<VkPhysicalDevice>& devices) {
        *observedDevices = vk::snapshotPhysicalDevices(funcs, devices);
        if (deviceObserver)
            deviceObserver(*observedDevices);

        for (size_t i = 0; i < devices.size(); ++i) {
            if (devicePicker(observedDevices->at(i).identity))
                return devices.at(i);
        }

        throw ls::vulkan_error("no suitable physical device found");
    };

    this->m_impl = std::make_unique<InstanceImpl>(
        selectFunc, observedDevices, shaderDllPath, allowLowPrecision
    );
}

namespace {
    /// find the cache file path
    std::filesystem::path findCacheFilePath() {
        const char* xdgCacheHome = std::getenv("XDG_CACHE_HOME");
        if (xdgCacheHome && *xdgCacheHome != '\0')
            return std::filesystem::path(xdgCacheHome) / "lsfg-vk_pipeline_cache.bin";

        const char* home = std::getenv("HOME");
        if (home && *home != '\0')
            return std::filesystem::path(home) / ".cache" / "lsfg-vk_pipeline_cache.bin";

        return{"/tmp/lsfg-vk_pipeline_cache.bin"};
    }
    /// create a Vulkan instance
    vk::Vulkan createVulkanInstance(vk::PhysicalDeviceSelector selectPhysicalDevice) {
        try {
            return{
                "lsfg-vk", vk::version{2, 0, 0},
                "lsfg-vk-engine", vk::version{2, 0, 0},
                selectPhysicalDevice,
                false, std::nullopt,
                findCacheFilePath()
            };
        } catch (const std::exception& e) {
            throw backend::error("Unable to initialize Vulkan", e);
        }
    }
    /// build a shader registry
    ShaderRegistry createShaderRegistry(vk::Vulkan& vk,
            const std::filesystem::path& shaderDllPath,
            bool allowLowPrecision) {
        std::unordered_map<uint32_t, std::vector<uint8_t>> resources{};

        try {
            resources = backend::extractResourcesFromDLL(shaderDllPath);
        } catch (const std::exception& e) {
            throw backend::error("Unable to parse Lossless Scaling DLL", e);
        }

        try {
            return backend::buildShaderRegistry(
                vk, allowLowPrecision && vk.supportsFP16(),
                resources
            );
        } catch (const std::exception& e) {
            throw backend::error("Unable to build shader registry", e);
        }
    }
#ifdef LSFGVK_TESTING_RENDERDOC
    /// load RenderDoc integration
    std::optional<RENDERDOC_API_1_6_0> loadRenderDocIntegration() {
        void* module = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (!module)
            return std::nullopt;

        auto renderdocGetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(
            dlsym(module, "RENDERDOC_GetAPI"));
        if (!renderdocGetAPI)
            return std::nullopt;

        RENDERDOC_API_1_6_0* api{};
        renderdocGetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void**>(&api));
        if (!api)
            return std::nullopt;

        return *api;
    }
#endif
    Ctx createCtx(const InstanceImpl& instance, VkExtent2D extent,
        bool hdr, float flow, bool perf, size_t count);
}

InstanceImpl::InstanceImpl(vk::PhysicalDeviceSelector selectPhysicalDevice,
            std::shared_ptr<std::vector<vk::PhysicalDeviceSnapshot>> observedDevices,
            const std::filesystem::path& shaderDllPath,
            bool allowLowPrecision)
        : vk(createVulkanInstance(selectPhysicalDevice)),
        visibleDevices(std::move(*observedDevices)),
        deviceIdentity(vk::getPhysicalDeviceIdentity(
            this->vk.fi(), this->vk.physdev())),
        shaders(createShaderRegistry(this->vk, shaderDllPath,
            allowLowPrecision && vk.supportsFP16())) {
#ifdef LSFGVK_TESTING_RENDERDOC
    this->renderdoc = loadRenderDocIntegration();
#endif
    vk.persistPipelineCache(); // will silently fail
}

const vk::PhysicalDeviceIdentity& Instance::deviceIdentity() const {
    return this->m_impl->getDeviceIdentity();
}

const std::vector<vk::PhysicalDeviceSnapshot>& Instance::visibleDevices() const {
    return this->m_impl->getVisibleDevices();
}

void Instance::validateRuntimePrepass(VkImage transportImage, VkExtent2D extent,
        VkFormat transportFormat, uint64_t transportModifier, float flow, bool perf) {
    if (transportFormat != VK_FORMAT_B8G8R8A8_UNORM)
        throw backend::error("P4C-D0 supports only the diagnostic B8G8R8A8 transport format");
    const auto& vk = this->m_impl->getVulkan();
    const auto family = vk.queueFamilyIndex();
    if (!vk.fi().GetPhysicalDeviceFormatProperties2)
        throw backend::error("P4C-D0 format-properties dispatch is unavailable");

    VkDrmFormatModifierProperties2EXT modifierProperties[32]{};
    VkDrmFormatModifierPropertiesList2EXT modifierList{
        VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT, nullptr,
        32, modifierProperties};
    VkFormatProperties2 transportProperties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        &modifierList, {}};
    vk.fi().GetPhysicalDeviceFormatProperties2(vk.physdev(), transportFormat,
        &transportProperties);
    bool blitSource = false;
    for (uint32_t i = 0; i < modifierList.drmFormatModifierCount; ++i) {
        if (modifierProperties[i].drmFormatModifier == transportModifier)
            blitSource = (modifierProperties[i].drmFormatModifierTilingFeatures
                & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
    }
    VkFormatProperties2 nativeProperties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
    vk.fi().GetPhysicalDeviceFormatProperties2(vk.physdev(), VK_FORMAT_R8G8B8A8_UNORM,
        &nativeProperties);
    const auto requiredNative = VK_FORMAT_FEATURE_BLIT_DST_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (!blitSource || (nativeProperties.formatProperties.optimalTilingFeatures
            & requiredNative) != requiredNative)
        throw backend::error("P4C-D0 backend format capability gate rejected B8->R8 blit");

    try {
        auto sources = std::pair<vk::Image, vk::Image>{
            vk::Image(vk, extent, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
            vk::Image(vk, extent, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)};
        auto ctx = createCtx(*this->m_impl, extent, false, flow, perf, 0);
        Mipmaps mipmaps(ctx, sources);
        std::array<Alpha0, 7> alpha0{
            Alpha0(ctx, mipmaps.getImages().at(0)), Alpha0(ctx, mipmaps.getImages().at(1)),
            Alpha0(ctx, mipmaps.getImages().at(2)), Alpha0(ctx, mipmaps.getImages().at(3)),
            Alpha0(ctx, mipmaps.getImages().at(4)), Alpha0(ctx, mipmaps.getImages().at(5)),
            Alpha0(ctx, mipmaps.getImages().at(6))};
        std::array<Alpha1, 7> alpha1{
            Alpha1(ctx, 3, alpha0.at(0).getImages()), Alpha1(ctx, 2, alpha0.at(1).getImages()),
            Alpha1(ctx, 2, alpha0.at(2).getImages()), Alpha1(ctx, 2, alpha0.at(3).getImages()),
            Alpha1(ctx, 2, alpha0.at(4).getImages()), Alpha1(ctx, 2, alpha0.at(5).getImages()),
            Alpha1(ctx, 2, alpha0.at(6).getImages())};
        Beta0 beta0(ctx, alpha1.at(0).getImages());
        Beta1 beta1(ctx, beta0.getImages());

        std::vector<VkImage> internal;
        mipmaps.prepare(internal);
        for (size_t i = 0; i < 7; ++i) {
            alpha0.at(i).prepare(internal);
            alpha1.at(i).prepare(internal);
        }
        beta0.prepare(internal);
        beta1.prepare(internal);

        vk::CommandBuffer command(vk);
        command.begin(vk);
        std::vector<vk::Barrier> barriers;
        barriers.reserve(3 + internal.size());
        barriers.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, family,
            transportImage, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
        for (const auto& source : {sources.first.handle(), sources.second.handle()})
            barriers.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED, source, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
        command.insertBarriers(vk, barriers, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkImageBlit blit{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {{0, 0, 0}, {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1}},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {{0, 0, 0}, {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1}}};
        vk.df().CmdBlitImage(command.handle(), transportImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sources.first.handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
        vk.df().CmdBlitImage(command.handle(), transportImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sources.second.handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

        std::vector<vk::Barrier> ready;
        ready.reserve(2 + internal.size());
        for (const auto& source : {sources.first.handle(), sources.second.handle()})
            ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, source,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
        for (const auto image : internal)
            ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, 0,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
        ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, family, VK_QUEUE_FAMILY_FOREIGN_EXT,
            transportImage, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
        command.insertBarriers(vk, ready, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        mipmaps.render(vk, command, 0);
        for (size_t i = 0; i < 7; ++i) {
            alpha0.at(6 - i).render(vk, command);
            alpha1.at(6 - i).render(vk, command, 0);
        }
        beta0.render(vk, command, 0);
        beta1.render(vk, command);
        command.end(vk);
        command.submit(vk);
        std::cerr << "[DG2X-P4C-D0] Runtime mode: CAPTURE_ONLY\n"
            << "  Transport B reacquire FOREIGN: PASS\n"
            << "  Backend input image creation: PASS\n"
            << "  B8 -> R8 GPU blit: PASS\n"
            << "  Source 0 seeded: PASS\n  Source 1 seeded: PASS\n"
            << "  LSFG mipmaps: SCHEDULED\n  LSFG Alpha0/Alpha1: SCHEDULED\n"
            << "  LSFG Beta0/Beta1: SCHEDULED\n  Generated frame count: 0\n"
            << "  Generate shader: NOT RUN\n  Destination images: NONE\n"
            << "  Backend submit: PASS\n  Backend fence completion: PASS\n"
            << "  CPU frame bridge: NONE\n"
            << "  Host wait before P4C-D0: YES (DIAGNOSTIC ONLY)\n"
            << "  LSFG backend execution: PREPASS_ONLY\n  Generated frame: NONE\n"
            << "  Presentation from B: NONE\n  Frame transport connected: INPUT_ONLY\n"
            << "DG2X_P4C_D0_LSFG_PREPASS_B_PASS\n";
    } catch (const std::exception& e) {
        throw backend::error("Unable to execute P4C-D0 LSFG prepass", e);
    }
}

vk::RuntimeExchangeEndpoint Instance::runtimeExchangeEndpoint() const {
    return vk::makeRuntimeExchangeEndpoint(this->m_impl->getVulkan());
}

Context& Instance::openContext(
        std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
        std::vector<vk::ExternalImage> destImages,
        int syncFd, float flow, bool perf) {
    const auto& descriptor = sourceImages.first.descriptor;
    const VkExtent2D extent{ descriptor.extent.width, descriptor.extent.height };
    const auto expectedSource = vk::makeSourceExchangeImageDescriptor(
        extent, descriptor.format);
    const auto expectedDestination = vk::makeDestinationExchangeImageDescriptor(
        extent, descriptor.format);

    if (!(sourceImages.first.descriptor == expectedSource)
            || !(sourceImages.second.descriptor == expectedSource))
        throw backend::error("Source exchange image descriptors do not match");
    for (const auto& image : destImages) {
        if (!(image.descriptor == expectedDestination))
            throw backend::error("Destination exchange image descriptors do not match");
    }

    const bool hdr = descriptor.format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!hdr && descriptor.format != VK_FORMAT_R8G8B8A8_UNORM)
        throw backend::error("Unsupported exchange image format");

    return *this->m_contexts.emplace_back(std::make_unique<ContextImpl>(*this->m_impl,
        std::move(sourceImages), std::move(destImages), syncFd,
        extent, hdr, flow, perf
    )).get();
}

namespace {
    /// import source images
    std::pair<vk::Image, vk::Image> importImages(const vk::Vulkan& vk,
            std::pair<vk::ExternalImage, vk::ExternalImage>& sourceImages) {
        try {
            return {
                vk::Image(vk, std::move(sourceImages.first)),
                vk::Image(vk, std::move(sourceImages.second))
            };
        } catch (const std::exception& e) {
            throw backend::error("Unable to import source images", e);
        }
    }
    /// import destination images
    std::vector<vk::Image> importImages(const vk::Vulkan& vk,
            std::vector<vk::ExternalImage>& externalImages) {
        try {
            std::vector<vk::Image> destImages;
            destImages.reserve(externalImages.size());

            for (auto& image : externalImages)
                destImages.emplace_back(vk, std::move(image));

            return destImages;
        } catch (const std::exception& e) {
            throw backend::error("Unable to import destination images", e);
        }
    }
    /// create a black image
    vk::Image createBlackImage(const vk::Vulkan& vk) {
        try {
            return{vk,
                { .width = 4, .height = 4 }
            };
        } catch (const std::exception& e) {
            throw backend::error("Unable to create black image", e);
        }
    }
    /// import timeline semaphore
    vk::TimelineSemaphore importTimelineSemaphore(const vk::Vulkan& vk, int syncFd) {
        try {
            return{vk, 0, syncFd};
        } catch (const std::exception& e) {
            throw backend::error("Unable to import timeline semaphore", e);
        }
    }
    /// create prepass semaphores
    vk::TimelineSemaphore createPrepassSemaphore(const vk::Vulkan& vk) {
        try {
            return{vk, 0};
        } catch (const std::exception& e) {
            throw backend::error("Unable to create prepass semaphore", e);
        }
    }
    /// create command buffers
    std::vector<vk::CommandBuffer> createCommandBuffers(const vk::Vulkan& vk, size_t count) {
        try {
            std::vector<vk::CommandBuffer> cmdbufs;
            cmdbufs.reserve(count);

            for (size_t i = 0; i < count; ++i)
                cmdbufs.emplace_back(vk);

            return cmdbufs;
        } catch (const std::exception& e) {
            throw backend::error("Unable to create command buffers", e);
        }
    }
    /// create context data
    Ctx createCtx(const InstanceImpl& instance, VkExtent2D extent,
            bool hdr, float flow, bool perf, size_t count) {
        const auto& vk = instance.getVulkan();
        const auto& shaders = instance.getShaderRegistry();

        try {
            std::vector<vk::Buffer> constantBuffers{};
            constantBuffers.reserve(count);

            for (size_t i = 0; i < count; ++i)
                constantBuffers.emplace_back(vk,
                    backend::getDefaultConstantBuffer(
                        i, count, flow
                    )
                );

            return {
                .vk = std::ref(vk),
                .shaders = std::ref(shaders),
                .pool{vk, backend::calculateDescriptorPoolLimits(count, perf)},
                .constantBuffer{vk, backend::getDefaultConstantBuffer(0, 1, flow)},
                .constantBuffers{std::move(constantBuffers)},
                .bnbSampler{vk, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_COMPARE_OP_NEVER, false},
                .bnwSampler{vk, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_COMPARE_OP_NEVER, true},
                .eabSampler{vk, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_COMPARE_OP_ALWAYS, false},
                .sourceExtent = extent,
                .flowExtent = VkExtent2D {
                    .width = static_cast<uint32_t>(static_cast<float>(extent.width) / flow),
                    .height = static_cast<uint32_t>(static_cast<float>(extent.height) / flow)
                },
                .hdr = hdr,
                .flow = flow,
                .perf = perf,
                .count = count
            };
        } catch (const std::exception& e) {
            throw backend::error("Unable to create context", e);
        }
    }
}

ContextImpl::ContextImpl(const InstanceImpl& instance,
            std::pair<vk::ExternalImage, vk::ExternalImage> externalSourceImages,
            std::vector<vk::ExternalImage> externalDestImages, int syncFd,
            VkExtent2D extent, bool hdr, float flow, bool perf) :
        sourceImages(importImages(instance.getVulkan(), externalSourceImages)),
        destImages(importImages(instance.getVulkan(), externalDestImages)),
        blackImage(createBlackImage(instance.getVulkan())),
        syncSemaphore(importTimelineSemaphore(instance.getVulkan(), syncFd)),
        prepassSemaphore(createPrepassSemaphore(instance.getVulkan())),
        destinationFirstUse(externalDestImages.size(), true),
        cmdbufs(createCommandBuffers(instance.getVulkan(), externalDestImages.size() + 1)),
        cmdbufFence(instance.getVulkan()),
        ctx(createCtx(instance, extent, hdr, flow, perf, externalDestImages.size())),
        mipmaps(ctx, sourceImages),
        alpha0{
            Alpha0(ctx, mipmaps.getImages().at(0)),
            Alpha0(ctx, mipmaps.getImages().at(1)),
            Alpha0(ctx, mipmaps.getImages().at(2)),
            Alpha0(ctx, mipmaps.getImages().at(3)),
            Alpha0(ctx, mipmaps.getImages().at(4)),
            Alpha0(ctx, mipmaps.getImages().at(5)),
            Alpha0(ctx, mipmaps.getImages().at(6))
        },
        alpha1{
            Alpha1(ctx, 3, alpha0.at(0).getImages()),
            Alpha1(ctx, 2, alpha0.at(1).getImages()),
            Alpha1(ctx, 2, alpha0.at(2).getImages()),
            Alpha1(ctx, 2, alpha0.at(3).getImages()),
            Alpha1(ctx, 2, alpha0.at(4).getImages()),
            Alpha1(ctx, 2, alpha0.at(5).getImages()),
            Alpha1(ctx, 2, alpha0.at(6).getImages())
        },
        beta0(ctx, alpha1.at(0).getImages()),
        beta1(ctx, beta0.getImages()) {
    // build main passes
    for (size_t i = 0; i < destImages.size(); ++i) {
        auto& pass = this->passes.emplace_back();

        pass.gamma0.reserve(7);
        pass.gamma1.reserve(7);
        pass.delta0.reserve(3);
        pass.delta1.reserve(3);
        for (size_t j = 0; j < 7; j++) {
            if (j == 0) { // first pass has no prior data
                pass.gamma0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    this->blackImage
                );
                pass.gamma1.emplace_back(ctx, i,
                    pass.gamma0.at(j).getImages(),
                    this->blackImage,
                    this->beta1.getImages().at(5)
                );
            } else { // other passes use prior data
                pass.gamma0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.gamma1.emplace_back(ctx, i,
                    pass.gamma0.at(j).getImages(),
                    pass.gamma1.at(j - 1).getImage(),
                    this->beta1.getImages().at(6 - j)
                );
            }

            if (j == 4) { // first special pass has no prior data
                pass.delta0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    this->blackImage,
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.delta1.emplace_back(ctx, i,
                    pass.delta0.at(j - 4).getImages0(),
                    pass.delta0.at(j - 4).getImages1(),
                    this->blackImage,
                    this->beta1.getImages().at(6 - j),
                    this->blackImage
                );
            } else if (j > 4) { // further passes do
                pass.delta0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    pass.delta1.at(j - 5).getImage0(),
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.delta1.emplace_back(ctx, i,
                    pass.delta0.at(j - 4).getImages0(),
                    pass.delta0.at(j - 4).getImages1(),
                    pass.delta1.at(j - 5).getImage0(),
                    this->beta1.getImages().at(6 - j),
                    pass.delta1.at(j - 5).getImage1()
                );
            }
        }

        pass.generate.emplace(ctx, i,
            this->sourceImages,
            pass.gamma1.at(6).getImage(),
            pass.delta1.at(2).getImage0(),
            pass.delta1.at(2).getImage1(),
            this->destImages.at(i)
        );
    }

    // initialize all images
    std::vector<VkImage> images{};
    images.push_back(this->blackImage.handle());
    mipmaps.prepare(images);
    for (size_t i = 0; i < 7; ++i) {
        alpha0.at(i).prepare(images);
        alpha1.at(i).prepare(images);
    }
    beta0.prepare(images);
    beta1.prepare(images);
    for (const auto& pass : this->passes) {
        for (size_t i = 0; i < 7; ++i) {
            pass.gamma0.at(i).prepare(images);
            pass.gamma1.at(i).prepare(images);

            if (i < 4) continue;
            pass.delta0.at(i - 4).prepare(images);
            pass.delta1.at(i - 4).prepare(images);
        }
    }

    std::vector<vk::Barrier> barriers{};
    barriers.reserve(images.size());

    for (const auto& image : images) {
        barriers.emplace_back(vk::Barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        });
    }

    const vk::CommandBuffer cmdbuf{ctx.vk};
    cmdbuf.begin(ctx.vk);
    cmdbuf.insertBarriers(ctx.vk, barriers);
    cmdbuf.end(ctx.vk);
    cmdbuf.submit(ctx.vk); // wait for completion
}

void Instance::scheduleFrames(
        Context& context,
        const std::vector<float>& timestamps) { // NOLINT (static)
#ifdef LSFGVK_TESTING_RENDERDOC
    const auto& impl = this->m_impl;
    if (impl->getRenderDocAPI()) {
        impl->getRenderDocAPI()->StartFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(impl->getVulkan().inst()),
            nullptr);
    }
#endif
    try {
        context.scheduleFrames(timestamps);
    } catch (const std::exception& e) {
        throw backend::error("Unable to schedule fixed-target frames", e);
    }
#ifdef LSFGVK_TESTING_RENDERDOC
    if (impl->getRenderDocAPI()) {
        impl->getVulkan().df().DeviceWaitIdle(impl->getVulkan().dev());
        impl->getRenderDocAPI()->EndFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(impl->getVulkan().inst()),
            nullptr);
    }
#endif
}

void Instance::scheduleFrames(Context& context) { // NOLINT (static)
#ifdef LSFGVK_TESTING_RENDERDOC
    const auto& impl = this->m_impl;
    if (impl->getRenderDocAPI()) {
        impl->getRenderDocAPI()->StartFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(impl->getVulkan().inst()),
            nullptr);
    }
#endif
    try {
        context.scheduleFrames();
    } catch (const std::exception& e) {
        throw backend::error("Unable to schedule frames", e);
    }
#ifdef LSFGVK_TESTING_RENDERDOC
    if (impl->getRenderDocAPI()) {
        impl->getVulkan().df().DeviceWaitIdle(impl->getVulkan().dev());
        impl->getRenderDocAPI()->EndFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(impl->getVulkan().inst()),
            nullptr);
    }
#endif
}

void Context::scheduleFrames() {
    if (this->fidx && !this->cmdbufFence.wait(this->ctx.vk))
        throw backend::error("Timeout waiting for previous frame to complete");
    this->cmdbufFence.reset(this->ctx.vk);
    this->schedulePreparedFrames(this->destImages.size());
}

void Context::schedulePreparedFrames(size_t generatedFrames) {
    const auto timeline = vk::makeExchangeTimelineFrame(this->idx, generatedFrames);
    const size_t currentSource = this->fidx % 2;
    const size_t returnedSource = (this->fidx + 1) % 2;
    const uint32_t family = this->ctx.vk.get().queueFamilyIndex();

    // schedule pre-pass
    const auto& cmdbuf = this->cmdbufs.at(0);
    cmdbuf.begin(ctx.vk);

    std::vector<vk::Barrier> sourceAcquires;
    if (this->fidx == 0) {
        sourceAcquires = {
            vk::sourceAcquireFromLayer(this->sourceImages.first.handle(), family),
            vk::sourceAcquireFromLayer(this->sourceImages.second.handle(), family)
        };
    } else {
        const auto& source = currentSource == 0
            ? this->sourceImages.first : this->sourceImages.second;
        sourceAcquires = { vk::sourceAcquireFromLayer(source.handle(), family) };
    }
    cmdbuf.insertBarriers(ctx.vk, sourceAcquires,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    this->mipmaps.render(ctx.vk, cmdbuf, this->fidx);
    for (size_t i = 0; i < 7; ++i) {
        this->alpha0.at(6 - i).render(ctx.vk, cmdbuf);
        this->alpha1.at(6 - i).render(ctx.vk, cmdbuf, this->fidx);
    }
    this->beta0.render(ctx.vk, cmdbuf, this->fidx);
    this->beta1.render(ctx.vk, cmdbuf);

    // Generate samples both alternating source images. With no Generate job,
    // queue order after this prepass is the last-read point for the previous
    // image; otherwise it is released by the final Generate command buffer.
    if (generatedFrames == 0) {
        const auto& source = returnedSource == 0
            ? this->sourceImages.first : this->sourceImages.second;
        cmdbuf.insertBarriers(ctx.vk,
            { vk::sourceReleaseToLayer(source.handle(), family) },
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    cmdbuf.end(ctx.vk);
    if (generatedFrames == 0) {
        cmdbuf.submit(this->ctx.vk,
            {}, this->syncSemaphore.handle(), timeline.sourceReady,
            {}, this->syncSemaphore.handle(), timeline.sourceReturn,
            this->cmdbufFence.handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    } else {
        cmdbuf.submit(this->ctx.vk,
            {}, this->syncSemaphore.handle(), timeline.sourceReady,
            {}, this->prepassSemaphore.handle(), timeline.sourceReady,
            VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }

    // schedule main passes
    for (size_t i = 0; i < generatedFrames; i++) {
        const auto& cmdbuf = this->cmdbufs.at(i + 1);
        cmdbuf.begin(ctx.vk);

        const auto destinationAcquire = this->destinationFirstUse.at(i)
            ? vk::destinationInitialAcquireFromLayer(this->destImages.at(i).handle(), family)
            : vk::destinationAcquireFromLayer(this->destImages.at(i).handle(), family);
        cmdbuf.insertBarriers(ctx.vk, { destinationAcquire },
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        const auto& pass = this->passes.at(i);
        for (size_t j = 0; j < 7; j++) {
            pass.gamma0.at(j).render(ctx.vk, cmdbuf, this->fidx);
            pass.gamma1.at(j).render(ctx.vk, cmdbuf);

            if (j < 4) continue;
            pass.delta0.at(j - 4).render(ctx.vk, cmdbuf, this->fidx);
            pass.delta1.at(j - 4).render(ctx.vk, cmdbuf);
        }
        pass.generate->render(ctx.vk, cmdbuf, this->fidx);

        std::vector<vk::Barrier> releases;
        if (i == generatedFrames - 1) {
            const auto& source = returnedSource == 0
                ? this->sourceImages.first : this->sourceImages.second;
            releases.push_back(vk::sourceReleaseToLayer(source.handle(), family));
        }
        releases.push_back(vk::destinationReleaseToLayer(
            this->destImages.at(i).handle(), family));
        cmdbuf.insertBarriers(ctx.vk, releases,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        cmdbuf.end(ctx.vk);
        cmdbuf.submit(this->ctx.vk,
            {}, this->prepassSemaphore.handle(), timeline.sourceReady,
            {}, this->syncSemaphore.handle(), timeline.destinationReady(i),
            i == generatedFrames - 1 ? this->cmdbufFence.handle() : VK_NULL_HANDLE,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
        );
        this->destinationFirstUse.at(i) = false;
    }

    this->idx = timeline.nextBase;
    this->fidx++;
}

void Context::scheduleFrames(const std::vector<float>& timestamps) {
    if (timestamps.size() > this->destImages.size())
        throw backend::error("requested more generated frames than the context capacity");

    float previous = 0.0F;
    for (const float timestamp : timestamps) {
        if (!(timestamp > 0.0F && timestamp < 1.0F))
            throw backend::error("frame generation timestamp must be between 0 and 1");
        if (timestamp <= previous)
            throw backend::error("frame generation timestamps must be strictly increasing");
        previous = timestamp;
    }

    // Wait for the previous source-frame processing to complete before updating
    // host-visible constant buffers that may still be referenced by the GPU.
    if (this->fidx && !this->cmdbufFence.wait(this->ctx.vk))
        throw backend::error("Timeout waiting for previous frame to complete");
    this->cmdbufFence.reset(this->ctx.vk);

    for (size_t i = 0; i < timestamps.size(); ++i) {
        auto constants = backend::getDefaultConstantBuffer(
            i, timestamps.size(), this->ctx.flow);
        constants.timestamp = timestamps.at(i);
        this->ctx.constantBuffers.at(i).update(this->ctx.vk, constants);
    }

    // The shared path keeps prepass/history processing active for g == 0.
    this->schedulePreparedFrames(timestamps.size());
}

void Instance::closeContext(const Context& context) {
    auto it = std::ranges::find_if(this->m_contexts,
        [context = &context](const std::unique_ptr<ContextImpl>& ctx) {
            return ctx.get() == context;
        });
    if (it == this->m_contexts.end())
        throw backend::error("attempted to close unknown context",
            std::runtime_error("no such context"));

    const auto& vk = this->m_impl->getVulkan();
    vk.df().DeviceWaitIdle(vk.dev());

    this->m_contexts.erase(it);
}

Instance::~Instance() = default;

// leaking shenanigans

namespace {
    bool leaking{false}; // NOLINT (global variable)
}

InstanceImpl::~InstanceImpl() {
    if (!leaking) return;

    try {
        new vk::Vulkan(std::move(this->vk));
    } catch (...) {
        std::cerr << "lsfg-vk: failed to leak Vulkan instance\n";
    }

}

void backend::makeLeaking() {
    leaking = true;
}
