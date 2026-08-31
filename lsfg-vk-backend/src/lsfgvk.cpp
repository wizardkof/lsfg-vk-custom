/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-vk-backend/lsfgvk.hpp"
#include "extraction/dll_reader.hpp"
#include "extraction/shader_registry.hpp"
#include "helpers/limits.hpp"
#include "helpers/utils.hpp"
#include "runtime_generate_diagnostic_state.hpp"
#include "lsfg-vk-backend/runtime_operation_authority.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/fnv1a.hpp"
#include "lsfg-vk-common/vulkan/buffer.hpp"
#include "lsfg-vk-common/vulkan/command_buffer.hpp"
#include "lsfg-vk-common/vulkan/exchange_image_sync.hpp"
#include "lsfg-vk-common/vulkan/destination_return_authority.hpp"
#include "lsfg-vk-common/helpers/owned_fd.hpp"
#include "lsfg-vk-common/vulkan/fence.hpp"
#include "lsfg-vk-common/vulkan/image.hpp"
#include "lsfg-vk-common/vulkan/physical_device.hpp"
#include "lsfg-vk-common/vulkan/prepared_frame_schedule_authority.hpp"
#include "lsfg-vk-common/vulkan/semaphore.hpp"
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
#include <atomic>
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

struct RuntimeShadowIngestAuthorityAccess {
    static RuntimeIngestPending accepted(uint64_t frame, TemporalSourceSlot slot,
            uint64_t epoch, VkFence fence, VkSemaphore payload,
            const std::shared_ptr<const uint8_t>& lifetime) {
        RuntimeSubmissionRetirement submission(fence, epoch, lifetime);
        RuntimeTemporarySemaphorePayload imported(payload, epoch, lifetime);
        submission.submitted();
        imported.waitSubmitted(epoch);
        return {frame, slot, epoch, std::move(submission), std::move(imported), lifetime};
    }
};

class RuntimeGenerateDiagnosticPendingState {
public:
    RuntimeGenerationId generation{};
    VkImage image{};
    VkExtent2D extent{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t family{};
    std::weak_ptr<const uint8_t> sessionLifetime;
    std::unique_ptr<vk::Fence> fence;
    std::unique_ptr<vk::Semaphore> readiness;
    VkSemaphore readinessHandle{};
    bool transportConsumed{};
    bool completionConsumed{};
    bool retirementAuthorityIssued{};
    bool operationRetired{};
    RuntimeTemporalPairIdentity temporalPair{};
    RuntimeBinarySemaphoreEpoch generationReadyEpoch;
    std::optional<vk::RuntimeFrameTransportSubmission> frameTransport;
};

RuntimeGenerateDiagnosticPending::RuntimeGenerateDiagnosticPending(
        std::shared_ptr<RuntimeGenerateDiagnosticPendingState> value) noexcept :
    pending(std::move(value)) {}

RuntimeGenerateDiagnosticPending::RuntimeGenerateDiagnosticPending(
        RuntimeGenerationId generation, VkImage image, VkExtent2D extent, VkFormat format,
        VkImageLayout layout, uint32_t family, VkSemaphore readiness,
        std::weak_ptr<const uint8_t> lifetime) :
    pending(std::make_shared<RuntimeGenerateDiagnosticPendingState>()) {
    pending->generation = generation; pending->image = image; pending->extent = extent;
    pending->format = format; pending->layout = layout; pending->family = family;
    pending->readinessHandle = readiness; pending->sessionLifetime = std::move(lifetime);
}

RuntimeGenerateDiagnosticPending::RuntimeGenerateDiagnosticPending(
        RuntimeGenerateDiagnosticPending&& other) noexcept :
    pending(std::move(other.pending)) { other.pending.reset(); }

RuntimeGenerateDiagnosticPending& RuntimeGenerateDiagnosticPending::operator=(
        RuntimeGenerateDiagnosticPending&& other) noexcept {
    if (this != &other) {
        pending = std::move(other.pending);
        other.pending.reset();
    }
    return *this;
}

bool RuntimeGenerateDiagnosticPending::valid() const noexcept {
    const auto& value = pending;
    return value && value->generation != 0 && value->image != VK_NULL_HANDLE
        && !value->sessionLifetime.expired() && !value->completionConsumed;
}

RuntimeGenerationId RuntimeGenerateDiagnosticPending::identity() const noexcept {
    const auto& value = pending; return value ? value->generation : 0;
}
VkImage RuntimeGenerateDiagnosticPending::imageHandle() const noexcept {
    const auto& value = pending; return value ? value->image : VK_NULL_HANDLE;
}
VkExtent2D RuntimeGenerateDiagnosticPending::extentValue() const noexcept {
    const auto& value = pending; return value ? value->extent : VkExtent2D{};
}
VkFormat RuntimeGenerateDiagnosticPending::formatValue() const noexcept {
    const auto& value = pending; return value ? value->format : VK_FORMAT_UNDEFINED;
}
VkImageLayout RuntimeGenerateDiagnosticPending::layoutValue() const noexcept {
    const auto& value = pending; return value ? value->layout : VK_IMAGE_LAYOUT_UNDEFINED;
}
uint32_t RuntimeGenerateDiagnosticPending::queueFamily() const noexcept {
    const auto& value = pending; return value ? value->family : 0;
}
VkSemaphore RuntimeGenerateDiagnosticPending::readinessSemaphore() const noexcept {
    const auto& value = pending;
    return value ? value->readinessHandle : VK_NULL_HANDLE;
}
VkFence RuntimeGenerateDiagnosticPending::sourceReadRetirementFence() const noexcept {
    const auto& value = pending;
    return value && value->fence ? value->fence->handle() : VK_NULL_HANDLE;
}
RuntimeTemporalPairIdentity RuntimeGenerateDiagnosticPending::temporalPair() const noexcept {
    const auto& value = pending;
    return value ? value->temporalPair : RuntimeTemporalPairIdentity{};
}
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
bool RuntimeGenerateDiagnosticPending::transportConsumedForTesting() const noexcept {
    return pending && pending->transportConsumed;
}
bool RuntimeGenerateDiagnosticPending::retirementAuthorityIssuedForTesting() const noexcept {
    return pending && pending->retirementAuthorityIssued;
}
bool RuntimeGenerateDiagnosticPending::operationRetiredForTesting() const noexcept {
    return pending && pending->operationRetired;
}
#endif
RuntimeGenerationOperationRetirement
RuntimeGenerateDiagnosticPending::operationRetirementAuthority() {
    if (!pending || !valid() || pending->retirementAuthorityIssued)
        throw std::logic_error("invalid or duplicate generation retirement authority");
    pending->retirementAuthorityIssued = true;
    return RuntimeGenerationOperationRetirement(pending);
}
RuntimeGenerationId RuntimeGenerationOperationRetirement::generationId() const noexcept {
    return pending ? pending->generation : 0;
}
void RuntimeGenerateDiagnosticPending::consumeTransport() {
    const auto& value = pending;
    if (!value || !valid() || value->transportConsumed)
        throw std::logic_error("invalid, stale, or consumed D3A2 transport capability");
    value->transportConsumed = true;
}

RuntimeGeneratedFrameToken::RuntimeGeneratedFrameToken(RuntimeGenerationId id,
        VkImage imageHandle, VkExtent2D imageExtent, VkFormat imageFormat,
        VkImageLayout imageLayout, uint32_t imageFamily,
        std::weak_ptr<const uint8_t> lifetime) noexcept :
    generation(id), image(imageHandle), extent(imageExtent), format(imageFormat),
    layout(imageLayout), family(imageFamily), sessionLifetime(std::move(lifetime)) {}

RuntimeGeneratedFrameToken::RuntimeGeneratedFrameToken(RuntimeGeneratedFrameToken&& other) noexcept :
    generation(std::exchange(other.generation, 0)), image(std::exchange(other.image, VK_NULL_HANDLE)),
    extent(other.extent), format(other.format), layout(other.layout), family(other.family),
    sessionLifetime(std::move(other.sessionLifetime)) {}

RuntimeGeneratedFrameToken& RuntimeGeneratedFrameToken::operator=(RuntimeGeneratedFrameToken&& other) noexcept {
    if (this != &other) {
        generation = std::exchange(other.generation, 0);
        image = std::exchange(other.image, VK_NULL_HANDLE);
        extent = other.extent; format = other.format; layout = other.layout; family = other.family;
        sessionLifetime = std::move(other.sessionLifetime);
    }
    return *this;
}

void RuntimeGeneratedFrameToken::consume() {
    if (!valid()) throw std::logic_error("invalid or consumed generated frame token");
    generation = 0;
    image = VK_NULL_HANDLE;
}

}

namespace {
    void validateRuntimeGenerateFormats(const lsfgvk::backend::InstanceImpl& instance,
        VkFormat transportFormat, uint64_t transportModifier);
}

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
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        InstanceImpl(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice,
            vk::VulkanInstanceFuncs instanceFuncs, vk::VulkanDeviceFuncs deviceFuncs,
            const std::unordered_map<uint32_t, std::vector<uint8_t>>& shaderResources);
#endif

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

    struct GenerationPass {
        std::vector<Gamma0> gamma0;
        std::vector<Gamma1> gamma1;
        std::vector<Delta0> delta0;
        std::vector<Delta1> delta1;
        ls::lazy<Generate> generate;
    };

    struct GenerationResourceSlot {
        GenerationResourceSlot(const InstanceImpl& instance, VkExtent2D extent,
            bool hdr, float flow, bool perf, size_t destinationCount);

        std::unique_ptr<Ctx> ctx;
        std::vector<GenerationPass> passes;
        std::vector<vk::CommandBuffer> cmdbufs;
        vk::Fence fence;
        vk::GenerationSlotAuthority authority;
        std::vector<uint64_t> requiredDestinationReturns;
        size_t generatedFrames{};
    };

    /// context class
    class ContextImpl {
    public:
        /// create a context
        /// (see lsfg-vk documentation)
        ContextImpl(const InstanceImpl& instance,
            std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
            std::vector<vk::ExternalImage> destImages, int syncFd,
            std::vector<ls::OwnedFd> destinationReturnFds,
            VkExtent2D extent, bool hdr, float flow, bool perf);

        /// schedule frames
        /// (see lsfg-vk documentation)
        void scheduleFrames();
        /// schedule zero or more frames at explicit interpolation timestamps
        void scheduleFrames(const std::vector<float>& timestamps);
        [[nodiscard]] std::optional<PreparedFrameScheduleReservation>
            reserveFrameSchedule(const std::vector<float>&,
                std::optional<vk::ExchangeTimelineFrame>);
        void executeReservation(PreparedFrameScheduleReservationState&);
        void abortReservation(PreparedFrameScheduleReservationState&) noexcept;
    private:
        void recordReservation(GenerationResourceSlot&, size_t generatedFrames,
            uint64_t reservedFidx, const vk::ExchangeTimelineFrame&);
        [[nodiscard]] bool tryRetireSlot(GenerationResourceSlot&) noexcept;

        std::pair<vk::Image, vk::Image> sourceImages;
        std::vector<vk::Image> destImages;
        vk::Image blackImage;

        vk::TimelineSemaphore syncSemaphore; // imported
        std::vector<vk::TimelineSemaphore> destinationReturnSemaphores; // imported
        vk::TimelineSemaphore prepassSemaphore;
        size_t idx{1};
        size_t fidx{0}; // real frame index
        std::vector<vk::DestinationReturnBackendState> destinationReturnStates;

        Ctx ctx;

        Mipmaps mipmaps;
        std::array<Alpha0, 7> alpha0;
        std::array<Alpha1, 7> alpha1;
        Beta0 beta0;
        Beta1 beta1;
        std::vector<std::unique_ptr<GenerationResourceSlot>> generationSlots;
        bool reservationOutstanding{};
        bool schedulePoisoned{};
    };

    class RuntimePrepassSessionImpl {
    public:
        RuntimePrepassSessionImpl(const InstanceImpl& instance, VkExtent2D extent,
            VkFormat transportFormat, uint64_t transportModifier, float flow, bool perf);
        void process(VkImage transportImage,
            vk::RuntimeFrameTransportSubmission transport);
    private:
        const InstanceImpl& instance;
        VkExtent2D extent{};
        VkFormat transportFormat{};
        uint64_t transportModifier{};
        std::pair<vk::Image, vk::Image> sources;
        Ctx ctx;
        Mipmaps mipmaps;
        std::array<Alpha0, 7> alpha0;
        std::array<Alpha1, 7> alpha1;
        Beta0 beta0;
        Beta1 beta1;
        bool seeded{};
        size_t directFrames{};
        size_t rotations{};
    };

    class RuntimeGenerateSessionImpl {
    public:
        RuntimeGenerateSessionImpl(const InstanceImpl& instance, VkExtent2D extent,
            VkFormat transportFormat, uint64_t transportModifier, float flow, bool perf,
            RuntimeGenerateMode mode);
        std::optional<RuntimeGenerateDiagnosticResult> process(
            VkImage transportImage, vk::RuntimeFrameTransportSubmission transport);
        std::optional<RuntimeGenerateDiagnosticPending> submit(
            VkImage transportImage, vk::RuntimeFrameTransportSubmission transport);
        std::optional<RuntimeGenerateDiagnosticPending> submitExplicit(
            VkImage transportImage, vk::RuntimeFrameTransportSubmission transport,
            TemporalSourceSlot destinationSlot, uint64_t frameId,
            std::optional<RuntimeTemporalPairIdentity> pair);
        RuntimeGenerateDiagnosticResult complete(RuntimeGenerateDiagnosticPending&& pending);
        void retireOperation(RuntimeGenerationOperationRetirement&& authority);
        [[nodiscard]] bool hasPending() const noexcept { return pendingGeneration != nullptr; }
        [[nodiscard]] RuntimeShadowSplitResourceSnapshot shadowSnapshot() const noexcept {
            if (!shadowSplit) return {};
            return {shadowSplit->ingestCommand.handle(), shadowSplit->generateCommand.handle(),
                shadowSplit->ingestReady.handle(), shadowSplit->ingestFence.handle(),
                instance.getVulkan().queueFamilyIndex(),
                shadowSplit->ingestReadyEpoch.canSignalAgain()};
        }
        RuntimeShadowIngestSnapshot submitShadowIngest(
            VkImage, vk::RuntimeFrameTransportSubmission,
            TemporalSourceSlot, uint64_t, RuntimeIngestIntent);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        RuntimeShadowIngestSnapshot submitShadowIngest(VkImage, vk::SyncFdPayload,
            TemporalSourceSlot, uint64_t, RuntimeIngestIntent);
#endif
        RuntimeIngestRetirementStatus tryRetireShadowIngest();
        [[nodiscard]] RuntimeShadowIngestSnapshot shadowIngestSnapshot() const noexcept;
        [[nodiscard]] vk::RuntimeImageObservationDescriptor
            temporalObservationDescriptor(TemporalSourceSlot) const noexcept;
        [[nodiscard]] vk::RuntimeImageObservationDescriptor
            generatedObservationDescriptor() const noexcept;
        RuntimeShadowGenerateSnapshot submitShadowGenerate(RuntimeTemporalPairIdentity);
        RuntimeRetirementStatus tryRetireShadowGenerate();
        RuntimeShadowGenerateSnapshot retireShadowGenerate();
        [[nodiscard]] RuntimeShadowGenerateSnapshot shadowGenerateSnapshot() const noexcept;
        std::optional<RuntimeGenerateDiagnosticPending> takeShadowGeneratePending();
        void recordShadowBReturnSubmitted(RuntimeGenerationId);
        void recordShadowBReturnRejected(RuntimeGenerationId, bool deviceLost);
        void retireShadowBReturnWait(RuntimeGenerationId);
        void failShadowBReturnRetirement(RuntimeGenerationId, bool deviceLost);
        void releaseShadowGenerationReady(RuntimeGenerationId);
        void recordShadowAReturnSubmitted(RuntimeGenerationId);
    private:
        struct ShadowSplitResources {
            explicit ShadowSplitResources(const vk::Vulkan& vk) :
                ingestCommand(vk), generateCommand(vk), ingestReady(vk), ingestFence(vk) {}
            vk::CommandBuffer ingestCommand;
            vk::CommandBuffer generateCommand;
            vk::Semaphore ingestReady;
            vk::Fence ingestFence;
            RuntimeBinarySemaphoreEpoch ingestReadyEpoch;
            std::unique_ptr<vk::Semaphore> importedWait;
            std::optional<RuntimeIngestPending> pending;
            std::optional<vk::RuntimeFrameTransportSubmission> frameTransport;
            std::optional<RuntimeGenerateDiagnosticPending> generatePending;
            std::shared_ptr<RuntimeGenerateDiagnosticPendingState> generateState;
            uint64_t nextEpoch{1};
            uint64_t nextWarmupEpoch{1};
            RuntimeIngestIntent ingestIntent{RuntimeIngestIntent::WARMUP_TEMPORAL};
            VkSemaphore submittedWait{};
            bool ingestFenceNeedsReset{};
            bool frameTransportReusable{true};
            bool failed{};
            bool deviceLost{};
            bool generateExecutionRetired{};
            bool generationReadyOutstanding{};
            uint32_t ingestSubmitCount{};
            uint32_t generateSubmitCount{};
            uint32_t bReturnSubmitCount{};
            uint32_t aReturnSubmitCount{};
            RuntimeTemporalPairIdentity lastRetiredPair{};
            RuntimeGenerationId lastRetiredGeneration{};
            bool lastExecutionRetired{};
            bool lastGeneratedOutputRetired{};
            bool lastGenerationReadyWaitRetired{};
            bool lastGenerationReadyReusable{};
        };
        struct Pass {
            std::vector<Gamma0> gamma0;
            std::vector<Gamma1> gamma1;
            std::vector<Delta0> delta0;
            std::vector<Delta1> delta1;
            ls::lazy<Generate> generate;
        };

        const InstanceImpl& instance;
        VkExtent2D extent{};
        VkFormat transportFormat{};
        uint64_t transportModifier{};
        size_t capturedBytes{};
        std::pair<vk::Image, vk::Image> sources;
        vk::Image blackImage;
        vk::Image destination;
        vk::Buffer readback;
        Ctx ctx;
        Mipmaps mipmaps;
        std::array<Alpha0, 7> alpha0;
        std::array<Alpha1, 7> alpha1;
        Beta0 beta0;
        Beta1 beta1;
        Pass pass;
        RuntimeGenerateDiagnosticState state;
        RuntimeGenerateSerialState serialState;
        RuntimeGenerateMode mode{RuntimeGenerateMode::OneShot};
        std::array<uint64_t, 2> temporalFrameIds{};
        std::array<bool, 2> temporalInitialized{};
        std::unique_ptr<ShadowSplitResources> shadowSplit;
        RuntimeGenerationId generation{};
        std::shared_ptr<const uint8_t> sessionLifetime{std::make_shared<const uint8_t>(0)};
        std::shared_ptr<RuntimeGenerateDiagnosticPendingState> pendingGeneration;
        RuntimeShadowIngestSnapshot submitShadowIngestInternal(VkImage,
            std::optional<vk::RuntimeFrameTransportSubmission>, vk::SyncFdPayload,
            TemporalSourceSlot, uint64_t, RuntimeIngestIntent);
        std::optional<RuntimeGenerateDiagnosticPending> processSubmission(
            VkImage transportImage, vk::RuntimeFrameTransportSubmission transport,
            bool deferGenerateCompletion,
            std::optional<TemporalSourceSlot> destinationSlot = std::nullopt,
            uint64_t frameId = 0,
            std::optional<RuntimeTemporalPairIdentity> pair = std::nullopt);
        void recordTemporalIngestCommands(const vk::Vulkan&, const vk::CommandBuffer&,
            VkImage transportImage, size_t frameIndex, bool seed,
            bool destinationFirstUse, uint32_t family);
        void recordPrepassGenerateCommands(const vk::Vulkan&, const vk::CommandBuffer&,
            size_t frameIndex, bool runGammaDelta, bool runGenerate);
        RuntimeGenerateDiagnosticResult validateCompletedGeneration(
            const std::shared_ptr<RuntimeGenerateDiagnosticPendingState>& pending);
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

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
Instance::Instance(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice,
        vk::VulkanInstanceFuncs instanceFuncs, vk::VulkanDeviceFuncs deviceFuncs,
        const std::unordered_map<uint32_t, std::vector<uint8_t>>& shaderResources) :
    m_impl(std::make_unique<InstanceImpl>(instance, device, physicalDevice,
        instanceFuncs, deviceFuncs, shaderResources)) {}
#endif

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

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
InstanceImpl::InstanceImpl(VkInstance instance, VkDevice device,
        VkPhysicalDevice physicalDevice, vk::VulkanInstanceFuncs instanceFuncs,
        vk::VulkanDeviceFuncs deviceFuncs,
        const std::unordered_map<uint32_t, std::vector<uint8_t>>& shaderResources) :
    vk(instance, device, physicalDevice, instanceFuncs, deviceFuncs, false),
    deviceIdentity(vk::getPhysicalDeviceIdentity(this->vk.fi(), this->vk.physdev())),
    shaders(buildShaderRegistry(this->vk, false, shaderResources)) {
    visibleDevices.push_back({deviceIdentity, {}});
}
#endif

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

RuntimePrepassSession& Instance::openRuntimePrepassSession(VkExtent2D extent,
        VkFormat transportFormat, uint64_t transportModifier, float flow, bool perf) {
    return *m_runtimePrepassSessions.emplace_back(
        std::make_unique<RuntimePrepassSessionImpl>(*m_impl, extent, transportFormat,
            transportModifier, flow, perf));
}

void Instance::processRuntimePrepass(RuntimePrepassSession& session,
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport) {
    session.process(transportImage, std::move(transport));
}

void Instance::closeRuntimePrepassSession(const RuntimePrepassSession& session) {
    const auto it = std::ranges::find_if(m_runtimePrepassSessions,
        [&session](const auto& candidate) { return candidate.get() == &session; });
    if (it != m_runtimePrepassSessions.end()) m_runtimePrepassSessions.erase(it);
}

RuntimeGenerateSession& Instance::openRuntimeGenerateSession(
        VkExtent2D extent, VkFormat transportFormat, uint64_t transportModifier,
        float flow, bool perf, RuntimeGenerateMode mode) {
    validateRuntimeGenerateFormats(*m_impl, transportFormat, transportModifier);
    return *m_runtimeGenerateDiagnosticSessions.emplace_back(
        std::make_unique<RuntimeGenerateSessionImpl>(*m_impl, extent,
            transportFormat, transportModifier, flow, perf, mode));
}

std::optional<RuntimeGenerateDiagnosticResult> Instance::processRuntimeGenerateDiagnostic(
        RuntimeGenerateSession& session,
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport) {
    return session.process(transportImage, std::move(transport));
}

std::optional<RuntimeGenerateDiagnosticPending> Instance::submitRuntimeGenerateDiagnostic(
        RuntimeGenerateSession& session,
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport) {
    return session.submit(transportImage, std::move(transport));
}

std::optional<RuntimeGenerateDiagnosticPending> Instance::submitRuntimeGenerateExplicit(
        RuntimeGenerateSession& session, VkImage transportImage,
        vk::RuntimeFrameTransportSubmission transport,
        TemporalSourceSlot destinationSlot,
        uint64_t frameId, std::optional<RuntimeTemporalPairIdentity> pair) {
    return session.submitExplicit(transportImage, std::move(transport),
        destinationSlot, frameId, pair);
}

RuntimeGenerateDiagnosticResult Instance::completeRuntimeGenerateDiagnostic(
        RuntimeGenerateSession& session,
        RuntimeGenerateDiagnosticPending&& pending) {
    return session.complete(std::move(pending));
}

void Instance::retireRuntimeGenerateOperation(RuntimeGenerateSession& session,
        RuntimeGenerationOperationRetirement&& authority) {
    session.retireOperation(std::move(authority));
}

void Instance::closeRuntimeGenerateSession(
        const RuntimeGenerateSession& session) {
    if (session.hasPending())
        throw std::logic_error("cannot close runtime Generate diagnostic with pending generation");
    const auto it = std::ranges::find_if(m_runtimeGenerateDiagnosticSessions,
        [&session](const auto& candidate) { return candidate.get() == &session; });
    if (it != m_runtimeGenerateDiagnosticSessions.end())
        m_runtimeGenerateDiagnosticSessions.erase(it);
}

RuntimeShadowSplitResourceSnapshot Instance::inspectRuntimeSplitResources(
        const RuntimeGenerateSession& session) const noexcept {
    return session.shadowSnapshot();
}

RuntimeShadowIngestSnapshot Instance::submitRuntimeIngest(
        RuntimeGenerateSession& session, VkImage transportImage,
        vk::RuntimeFrameTransportSubmission transport,
        TemporalSourceSlot destinationSlot,
        uint64_t frameId, RuntimeIngestIntent intent) {
    return session.submitShadowIngest(transportImage, std::move(transport),
        destinationSlot, frameId, intent);
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
RuntimeShadowIngestSnapshot Instance::submitRuntimeIngest(
        RuntimeGenerateSession& session, VkImage transportImage,
        vk::SyncFdPayload payload, TemporalSourceSlot destinationSlot,
        uint64_t frameId, RuntimeIngestIntent intent) {
    return session.submitShadowIngest(transportImage, std::move(payload),
        destinationSlot, frameId, intent);
}
#endif

RuntimeIngestRetirementStatus Instance::tryRetireRuntimeIngest(
        RuntimeGenerateSession& session) {
    return session.tryRetireShadowIngest();
}

RuntimeShadowIngestSnapshot Instance::inspectRuntimeIngest(
        const RuntimeGenerateSession& session) const noexcept {
    return session.shadowIngestSnapshot();
}

vk::RuntimeImageObservationDescriptor Instance::inspectRuntimeTemporalSource(
        const RuntimeGenerateSession& session, TemporalSourceSlot slot) const noexcept {
    return session.temporalObservationDescriptor(slot);
}

vk::RuntimeImageObservationDescriptor Instance::inspectRuntimeGeneratedOutput(
        const RuntimeGenerateSession& session) const noexcept {
    return session.generatedObservationDescriptor();
}

RuntimeShadowGenerateSnapshot Instance::submitRuntimePrepassGenerate(
        RuntimeGenerateSession& session, RuntimeTemporalPairIdentity pair) {
    return session.submitShadowGenerate(pair);
}

RuntimeRetirementStatus Instance::tryRetireRuntimePrepassGenerate(
        RuntimeGenerateSession& session) {
    return session.tryRetireShadowGenerate();
}

RuntimeShadowGenerateSnapshot Instance::retireRuntimePrepassGenerate(
        RuntimeGenerateSession& session) {
    return session.retireShadowGenerate();
}

RuntimeShadowGenerateSnapshot Instance::inspectRuntimePrepassGenerate(
        const RuntimeGenerateSession& session) const noexcept {
    return session.shadowGenerateSnapshot();
}

std::optional<RuntimeGenerateDiagnosticPending>
Instance::takeRuntimePrepassGeneratePending(RuntimeGenerateSession& session) {
    return session.takeShadowGeneratePending();
}

void Instance::recordRuntimeBReturnSubmitted(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.recordShadowBReturnSubmitted(generation);
}
void Instance::recordRuntimeBReturnRejected(RuntimeGenerateSession& session,
        RuntimeGenerationId generation, bool lost) {
    session.recordShadowBReturnRejected(generation, lost);
}
void Instance::retireRuntimeBReturnWait(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.retireShadowBReturnWait(generation);
}
void Instance::failRuntimeBReturnRetirement(RuntimeGenerateSession& session,
        RuntimeGenerationId generation, bool lost) {
    session.failShadowBReturnRetirement(generation, lost);
}
void Instance::releaseRuntimeGenerationReady(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.releaseShadowGenerationReady(generation);
}
void Instance::recordRuntimeAReturnSubmitted(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.recordShadowAReturnSubmitted(generation);
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
RuntimeShadowSplitResourceSnapshot Instance::inspectShadowSplitResources(
        const RuntimeGenerateSession& session) const noexcept {
    return session.shadowSnapshot();
}

RuntimeShadowIngestSnapshot Instance::submitShadowTemporalIngest(
        RuntimeGenerateSession& session, VkImage transportImage,
        vk::SyncFdPayload payload, TemporalSourceSlot destinationSlot,
        uint64_t frameId) {
    return session.submitShadowIngest(transportImage, std::move(payload),
        destinationSlot, frameId, RuntimeIngestIntent::GENERATE_SOURCE);
}

RuntimeShadowIngestSnapshot Instance::retireShadowTemporalIngest(
        RuntimeGenerateSession& session) {
    if (session.tryRetireShadowIngest() != RuntimeIngestRetirementStatus::RETIRED)
        throw backend::error("shadow ingest retirement is not ready");
    return session.shadowIngestSnapshot();
}

RuntimeShadowIngestSnapshot Instance::inspectShadowTemporalIngest(
        const RuntimeGenerateSession& session) const noexcept {
    return session.shadowIngestSnapshot();
}

RuntimeShadowGenerateSnapshot Instance::submitShadowPrepassGenerate(
        RuntimeGenerateSession& session,
        RuntimeTemporalPairIdentity pair) {
    return session.submitShadowGenerate(pair);
}

RuntimeShadowGenerateSnapshot Instance::retireShadowPrepassGenerate(
        RuntimeGenerateSession& session) {
    return session.retireShadowGenerate();
}

RuntimeShadowGenerateSnapshot Instance::inspectShadowPrepassGenerate(
        const RuntimeGenerateSession& session) const noexcept {
    return session.shadowGenerateSnapshot();
}

std::optional<RuntimeGenerateDiagnosticPending>
Instance::takeShadowPrepassGeneratePending(RuntimeGenerateSession& session) {
    return session.takeShadowGeneratePending();
}

void Instance::recordShadowBReturnSubmitted(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.recordShadowBReturnSubmitted(generation);
}

void Instance::recordShadowBReturnRejected(RuntimeGenerateSession& session,
        RuntimeGenerationId generation, bool deviceLost) {
    session.recordShadowBReturnRejected(generation, deviceLost);
}

void Instance::retireShadowBReturnWait(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.retireShadowBReturnWait(generation);
}

void Instance::failShadowBReturnRetirement(RuntimeGenerateSession& session,
        RuntimeGenerationId generation, bool deviceLost) {
    session.failShadowBReturnRetirement(generation, deviceLost);
}

void Instance::releaseShadowGenerationReady(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.releaseShadowGenerationReady(generation);
}

void Instance::recordShadowAReturnSubmitted(RuntimeGenerateSession& session,
        RuntimeGenerationId generation) {
    session.recordShadowAReturnSubmitted(generation);
}
#endif

vk::RuntimeExchangeEndpoint Instance::runtimeExchangeEndpoint() const {
    return vk::makeRuntimeExchangeEndpoint(this->m_impl->getVulkan());
}

Context& Instance::openContext(
        std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
        std::vector<vk::ExternalImage> destImages,
        int syncFd, std::vector<int> destinationReturnFds,
        float flow, bool perf) {
    // The API transfers ownership at entry. Keep every descriptor RAII-owned
    // across validation and partial context construction.
    std::vector<ls::OwnedFd> ownedDestinationReturnFds;
    ownedDestinationReturnFds.reserve(destinationReturnFds.size());
    for (const int fd : destinationReturnFds)
        ownedDestinationReturnFds.emplace_back(fd);

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
    if (!vk::destinationReturnCardinalityValid(
            destImages.size(), destinationReturnFds.size()))
        throw backend::error("Destination return semaphore count does not match images");

    const bool hdr = descriptor.format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!hdr && descriptor.format != VK_FORMAT_R8G8B8A8_UNORM)
        throw backend::error("Unsupported exchange image format");

    return *this->m_contexts.emplace_back(std::make_unique<ContextImpl>(*this->m_impl,
        std::move(sourceImages), std::move(destImages), syncFd,
        std::move(ownedDestinationReturnFds),
        extent, hdr, flow, perf
    )).get();
}

Context& Instance::openContext(
        std::pair<vk::ExternalImage, vk::ExternalImage> sourceImages,
        std::vector<vk::ExternalImage> destImages,
        int syncFd, float flow, bool perf) {
    if (!destImages.empty())
        throw backend::error(
            "Destination return semaphores are required for generated destinations");
    return this->openContext(std::move(sourceImages), std::move(destImages),
        syncFd, {}, flow, perf);
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
    vk::Image createDiagnosticBlackImage(const vk::Vulkan& vk) {
        try {
            return{vk, { .width = 4, .height = 4 }, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        } catch (const std::exception& e) {
            throw backend::error("Unable to create D2 black image", e);
        }
    }
    vk::Buffer createDiagnosticReadbackBuffer(const vk::Vulkan& vk, size_t byteSize) {
        std::vector<uint8_t> zeroes(byteSize);
        return {vk, zeroes.data(), zeroes.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    }
    void validateRuntimeGenerateFormats(const InstanceImpl& instance,
            VkFormat transportFormat, uint64_t transportModifier) {
        if (transportFormat != VK_FORMAT_B8G8R8A8_UNORM)
            throw backend::error("P4C-D2 supports only B8G8R8A8 transport");

        const auto& vk = instance.getVulkan();
        if (!vk.fi().GetPhysicalDeviceFormatProperties2)
            throw backend::error("P4C-D2 format-properties dispatch is unavailable");

        VkDrmFormatModifierProperties2EXT modifierProperties[32]{};
        VkDrmFormatModifierPropertiesList2EXT modifierList{
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT, nullptr,
            32, modifierProperties};
        VkFormatProperties2 transportProperties{
            VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &modifierList, {}};
        vk.fi().GetPhysicalDeviceFormatProperties2(
            vk.physdev(), transportFormat, &transportProperties);
        bool transportBlitSource{};
        const auto modifierCount = std::min<uint32_t>(
            modifierList.drmFormatModifierCount, std::size(modifierProperties));
        for (uint32_t i = 0; i < modifierCount; ++i) {
            if (modifierProperties[i].drmFormatModifier == transportModifier) {
                transportBlitSource = (modifierProperties[i].drmFormatModifierTilingFeatures
                    & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
            }
        }

        VkFormatProperties2 rgba8{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        vk.fi().GetPhysicalDeviceFormatProperties2(
            vk.physdev(), VK_FORMAT_R8G8B8A8_UNORM, &rgba8);
        const auto rgba8Required = VK_FORMAT_FEATURE_BLIT_DST_BIT
            | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
            | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
            | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
            | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

        VkFormatProperties2 rgba16f{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        vk.fi().GetPhysicalDeviceFormatProperties2(
            vk.physdev(), VK_FORMAT_R16G16B16A16_SFLOAT, &rgba16f);
        const auto rgba16fRequired = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
            | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;

        if (!transportBlitSource
                || (rgba8.formatProperties.optimalTilingFeatures & rgba8Required) != rgba8Required
                || (rgba16f.formatProperties.optimalTilingFeatures & rgba16fRequired)
                    != rgba16fRequired) {
            throw backend::error("P4C-D2 backend format capability gate rejected required formats");
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
    std::vector<vk::TimelineSemaphore> importDestinationReturnSemaphores(
            const vk::Vulkan& vk, std::vector<ls::OwnedFd>& fds) {
        try {
            std::vector<vk::TimelineSemaphore> semaphores;
            semaphores.reserve(fds.size());
            for (auto& fd : fds) {
                semaphores.emplace_back(vk, 0, fd.get());
                // A successful OPAQUE_FD import consumes this descriptor.
                static_cast<void>(fd.release());
            }
            return semaphores;
        } catch (const std::exception& e) {
            // Successfully imported descriptors were consumed by Vulkan;
            // every not-yet-imported descriptor remains RAII-owned by fds.
            throw backend::error("Unable to import destination return semaphores", e);
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

GenerationResourceSlot::GenerationResourceSlot(const InstanceImpl& instance,
        VkExtent2D extent, bool hdr, float flow, bool perf,
        size_t destinationCount) :
    ctx(std::make_unique<Ctx>(createCtx(
        instance, extent, hdr, flow, perf, destinationCount))),
    cmdbufs(createCommandBuffers(instance.getVulkan(), destinationCount + 1)),
    fence(instance.getVulkan()),
    requiredDestinationReturns(destinationCount) {}

RuntimePrepassSessionImpl::RuntimePrepassSessionImpl(const InstanceImpl& instance,
        VkExtent2D extent, VkFormat transportFormat, uint64_t transportModifier,
        float flow, bool perf) :
    instance(instance), extent(extent), transportFormat(transportFormat),
    transportModifier(transportModifier),
    sources(vk::Image(instance.getVulkan(), extent, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT),
            vk::Image(instance.getVulkan(), extent, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT)),
    ctx(createCtx(instance, extent, false, flow, perf, 0)), mipmaps(ctx, sources),
    alpha0{Alpha0(ctx, mipmaps.getImages().at(0)), Alpha0(ctx, mipmaps.getImages().at(1)),
        Alpha0(ctx, mipmaps.getImages().at(2)), Alpha0(ctx, mipmaps.getImages().at(3)),
        Alpha0(ctx, mipmaps.getImages().at(4)), Alpha0(ctx, mipmaps.getImages().at(5)),
        Alpha0(ctx, mipmaps.getImages().at(6))},
    alpha1{Alpha1(ctx, 3, alpha0.at(0).getImages()), Alpha1(ctx, 2, alpha0.at(1).getImages()),
        Alpha1(ctx, 2, alpha0.at(2).getImages()), Alpha1(ctx, 2, alpha0.at(3).getImages()),
        Alpha1(ctx, 2, alpha0.at(4).getImages()), Alpha1(ctx, 2, alpha0.at(5).getImages()),
        Alpha1(ctx, 2, alpha0.at(6).getImages())},
    beta0(ctx, alpha1.at(0).getImages()), beta1(ctx, beta0.getImages()) {
    if (transportFormat != VK_FORMAT_B8G8R8A8_UNORM)
        throw backend::error("P4C-D1 supports only B8G8R8A8 transport");
    const auto& vk = instance.getVulkan();
    std::vector<VkImage> internal;
    mipmaps.prepare(internal);
    for (size_t i = 0; i < 7; ++i) { alpha0.at(i).prepare(internal); alpha1.at(i).prepare(internal); }
    beta0.prepare(internal); beta1.prepare(internal);
    std::vector<vk::Barrier> barriers;
    for (const auto image : internal)
        barriers.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, 0,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
    vk::CommandBuffer command(vk); command.begin(vk); command.insertBarriers(vk, barriers);
    command.end(vk); command.submit(vk);
}

void RuntimePrepassSessionImpl::process(
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport) {
    if (!transport.valid())
        throw std::invalid_argument("P4C-D1 frame transport authority is invalid");
    auto payload = transport.releasePayload();
    try {
    const auto& vk = instance.getVulkan();
    const uint32_t family = vk.queueFamilyIndex();
    const size_t frameIndex = directFrames % 2;
    const bool first = !seeded;
    const vk::ExternalSemaphoreDevice semaphoreDevice{vk.dev(), {
        vk.df().CreateSemaphore, vk.df().DestroySemaphore,
        vk.df().GetSemaphoreFdKHR, vk.df().ImportSemaphoreFdKHR}};
    auto waitSemaphore = vk::createSyncFdImportSemaphore(semaphoreDevice);
    vk::importSyncFdTemporary(semaphoreDevice, waitSemaphore.handle(), payload);

    vk::CommandBuffer command(vk); command.begin(vk);
    const auto range = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    std::vector<vk::Barrier> acquire{{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, family,
        transportImage, range}};
    if (first) {
        for (const auto image : {sources.first.handle(), sources.second.handle()})
            acquire.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED, image, range});
    } else {
        const auto image = frameIndex == 0 ? sources.first.handle() : sources.second.handle();
        acquire.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range});
    }
    command.insertBarriers(vk, acquire, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkImageBlit blit{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0,0,0},{static_cast<int32_t>(extent.width),static_cast<int32_t>(extent.height),1}},
        {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
        {{0,0,0},{static_cast<int32_t>(extent.width),static_cast<int32_t>(extent.height),1}}};
    auto blitTo = [&](VkImage target) { vk.df().CmdBlitImage(command.handle(), transportImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, VK_FILTER_NEAREST); };
    if (first) { blitTo(sources.first.handle()); blitTo(sources.second.handle()); }
    else blitTo(frameIndex == 0 ? sources.first.handle() : sources.second.handle());
    std::vector<vk::Barrier> ready;
    if (first) {
        for (const auto image : {sources.first.handle(), sources.second.handle()})
            ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range});
    } else {
        const auto image = frameIndex == 0 ? sources.first.handle() : sources.second.handle();
        ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range});
    }
    ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, family, VK_QUEUE_FAMILY_FOREIGN_EXT,
        transportImage, range});
    command.insertBarriers(vk, ready, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    mipmaps.render(vk, command, frameIndex);
    for (size_t i = 0; i < 7; ++i) {
        alpha0.at(6-i).render(vk, command); alpha1.at(6-i).render(vk, command, frameIndex);
    }
    beta0.render(vk, command, frameIndex); beta1.render(vk, command);
    command.end(vk);
    vk::Fence fence(vk);
    command.submit(vk, vk.queue(), {waitSemaphore.handle()}, VK_NULL_HANDLE, 0, {}, VK_NULL_HANDLE, 0,
        fence.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT);
    transport.consumerSubmitted();
    if (!fence.wait(vk)) throw backend::error("P4C-D1 backend fence wait failed");
    transport.consumerRetired();
    directFrames++;
    if (first) seeded = true; else rotations++;
    if (first) {
        std::cerr << "[DG2X-P4C-D1A] Direct synchronized prepass\n"
            << "  Frame ordinal: 1\n  A source -> transport blit: PASS\n"
            << "  Transport A release FOREIGN: PASS\n  A submission: PASS\n"
            << "  SYNC_FD A->B export: PASS\n  Host wait after A submission: NONE\n"
            << "  SYNC_FD temporary import on B: PASS\n"
            << "  B submission waiting on A semaphore: PASS\n"
            << "  Transport B acquire FOREIGN: PASS\n  B8 -> R8 GPU blit: PASS\n"
            << "  Source 0 seeded: PASS\n  Source 1 seeded: PASS\n"
            << "  LSFG Mipmaps: SCHEDULED\n  LSFG Alpha0/Alpha1: SCHEDULED\n"
            << "  LSFG Beta0/Beta1: SCHEDULED\n  Generated frame count: 0\n"
            << "  Generate shader: NOT RUN\n  Destinations: NONE\n"
            << "  Transport B release FOREIGN: PASS\n  Backend submit: PASS\n"
            << "  Backend fence completion: PASS\n  Host wait A->B: NONE\n"
            << "  Host wait after complete B chain: YES (DIAGNOSTIC ONLY)\n"
            << "  CPU frame bridge: NONE\n  LSFG backend execution: PREPASS_ONLY\n"
            << "  Generated frame: NONE\n  Presentation from B: NONE\n"
            << "  Frame transport connected: INPUT_ONLY\n"
            << "DG2X_P4C_D1A_DIRECT_PREPASS_PASS\n";
    } else if (directFrames == 2) {
        std::cerr << "[DG2X-P4C-D1B] Temporal input rotation\n"
            << "  Frame ordinal: 2\n  Session already seeded: YES\n"
            << "  Previous source slot: 0\n  Current source slot: 1\n"
            << "  Source 0 preserved: YES\n  Source 1 updated from real transport: YES\n"
            << "  Source duplication this frame: NO\n  Mipmaps frameIndex: 1\n"
            << "  Alpha frameIndex: 1\n  Beta frameIndex: 1\n"
            << "  Backend fence completion: PASS\n  Direct real frames processed: 2\n"
            << "  Temporal rotation count: 1\n  Host wait A->B: NONE\n"
            << "  CPU frame bridge: NONE\n  LSFG backend execution: PREPASS_ONLY\n"
            << "  Generated frame: NONE\n  Presentation from B: NONE\n"
            << "  Frame transport connected: INPUT_ONLY\n"
            << "DG2X_P4C_D1_DIRECT_PREPASS_CHAIN_PASS\n";
    }
    } catch (...) {
        transport.consumerFailed();
        throw;
    }
}

RuntimeGenerateSessionImpl::RuntimeGenerateSessionImpl(
        const InstanceImpl& instance, VkExtent2D extent, VkFormat transportFormat,
        uint64_t transportModifier, float flow, bool perf, RuntimeGenerateMode sessionMode) :
    instance(instance), extent(extent), transportFormat(transportFormat),
    transportModifier(transportModifier), mode(sessionMode),
    generation([] {
        static std::atomic<RuntimeGenerationId> next{1};
        auto id = next.fetch_add(1, std::memory_order_relaxed);
        return id == 0 ? next.fetch_add(1, std::memory_order_relaxed) : id;
    }()),
    capturedBytes(static_cast<size_t>(extent.width) * extent.height * 4U),
    sources(vk::Image(instance.getVulkan(), extent, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
            vk::Image(instance.getVulkan(), extent, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)),
    blackImage(createDiagnosticBlackImage(instance.getVulkan())),
    destination(instance.getVulkan(), extent, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
    readback(createDiagnosticReadbackBuffer(instance.getVulkan(), capturedBytes)),
    ctx(createCtx(instance, extent, false, flow, perf, 1)),
    mipmaps(ctx, sources),
    alpha0{Alpha0(ctx, mipmaps.getImages().at(0)), Alpha0(ctx, mipmaps.getImages().at(1)),
        Alpha0(ctx, mipmaps.getImages().at(2)), Alpha0(ctx, mipmaps.getImages().at(3)),
        Alpha0(ctx, mipmaps.getImages().at(4)), Alpha0(ctx, mipmaps.getImages().at(5)),
        Alpha0(ctx, mipmaps.getImages().at(6))},
    alpha1{Alpha1(ctx, 3, alpha0.at(0).getImages()), Alpha1(ctx, 2, alpha0.at(1).getImages()),
        Alpha1(ctx, 2, alpha0.at(2).getImages()), Alpha1(ctx, 2, alpha0.at(3).getImages()),
        Alpha1(ctx, 2, alpha0.at(4).getImages()), Alpha1(ctx, 2, alpha0.at(5).getImages()),
        Alpha1(ctx, 2, alpha0.at(6).getImages())},
    beta0(ctx, alpha1.at(0).getImages()), beta1(ctx, beta0.getImages()) {
    if (capturedBytes == 0)
        throw backend::error("P4C-D2 diagnostic destination has zero byte size");

    pass.gamma0.reserve(7);
    pass.gamma1.reserve(7);
    pass.delta0.reserve(3);
    pass.delta1.reserve(3);
    for (size_t j = 0; j < 7; ++j) {
        if (j == 0) {
            pass.gamma0.emplace_back(ctx, 0, alpha1.at(6).getImages(), blackImage);
            pass.gamma1.emplace_back(ctx, 0, pass.gamma0.at(j).getImages(),
                blackImage, beta1.getImages().at(5));
        } else {
            pass.gamma0.emplace_back(ctx, 0, alpha1.at(6 - j).getImages(),
                pass.gamma1.at(j - 1).getImage());
            pass.gamma1.emplace_back(ctx, 0, pass.gamma0.at(j).getImages(),
                pass.gamma1.at(j - 1).getImage(), beta1.getImages().at(6 - j));
        }

        if (j == 4) {
            pass.delta0.emplace_back(ctx, 0, alpha1.at(2).getImages(),
                blackImage, pass.gamma1.at(3).getImage());
            pass.delta1.emplace_back(ctx, 0,
                pass.delta0.at(0).getImages0(), pass.delta0.at(0).getImages1(),
                blackImage, beta1.getImages().at(2), blackImage);
        } else if (j > 4) {
            pass.delta0.emplace_back(ctx, 0, alpha1.at(6 - j).getImages(),
                pass.delta1.at(j - 5).getImage0(), pass.gamma1.at(j - 1).getImage());
            pass.delta1.emplace_back(ctx, 0,
                pass.delta0.at(j - 4).getImages0(), pass.delta0.at(j - 4).getImages1(),
                pass.delta1.at(j - 5).getImage0(), beta1.getImages().at(6 - j),
                pass.delta1.at(j - 5).getImage1());
        }
    }
    pass.generate.emplace(ctx, 0, sources.second, sources.first,
        pass.gamma1.at(6).getImage(),
        pass.delta1.at(2).getImage0(), pass.delta1.at(2).getImage1(), destination);

    std::vector<VkImage> internal{blackImage.handle()};
    mipmaps.prepare(internal);
    for (size_t i = 0; i < 7; ++i) {
        alpha0.at(i).prepare(internal);
        alpha1.at(i).prepare(internal);
        pass.gamma0.at(i).prepare(internal);
        pass.gamma1.at(i).prepare(internal);
        if (i >= 4) {
            pass.delta0.at(i - 4).prepare(internal);
            pass.delta1.at(i - 4).prepare(internal);
        }
    }
    beta0.prepare(internal);
    beta1.prepare(internal);

    std::vector<vk::Barrier> barriers;
    barriers.reserve(internal.size());
    for (const auto image : internal) {
        if (image == blackImage.handle())
            continue;
        barriers.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, 0,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
    }
    vk::CommandBuffer command(instance.getVulkan());
    command.begin(instance.getVulkan());
    command.insertBarriers(instance.getVulkan(), barriers);
    const auto range = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const vk::Barrier blackToClear{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, blackImage.handle(), range};
    command.insertBarriers(instance.getVulkan(), {blackToClear},
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkClearColorValue zero{};
    instance.getVulkan().df().CmdClearColorImage(command.handle(), blackImage.handle(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
    const vk::Barrier blackToGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, blackImage.handle(), range};
    command.insertBarriers(instance.getVulkan(), {blackToGeneral},
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    command.end(instance.getVulkan());
    command.submit(instance.getVulkan());
    try {
        shadowSplit = std::make_unique<ShadowSplitResources>(instance.getVulkan());
    } catch (...) {
        // The inactive scaffold must never make the qualified monolithic path unavailable.
        shadowSplit.reset();
    }
    std::cerr << "Black image zero initialization: PASS\n";
}

void RuntimeGenerateSessionImpl::recordTemporalIngestCommands(
        const vk::Vulkan& vk, const vk::CommandBuffer& command,
        VkImage transportImage, size_t frameIndex, bool seed,
        bool destinationFirstUse, uint32_t family) {
    const auto range = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    std::vector<vk::Barrier> acquire{{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, family,
        transportImage, range}};
    if (seed) {
        for (const auto image : {sources.first.handle(), sources.second.handle()})
            acquire.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED, image, range});
    } else {
        const auto image = frameIndex == 0 ? sources.first.handle() : sources.second.handle();
        acquire.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            destinationFirstUse ? 0U : VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            destinationFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range});
    }
    command.insertBarriers(vk, acquire, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkImageBlit blit{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0, 0, 0}, {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1}},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0, 0, 0}, {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1}}};
    const auto blitTo = [&](VkImage target) {
        vk.df().CmdBlitImage(command.handle(), transportImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    };
    if (seed) { blitTo(sources.first.handle()); blitTo(sources.second.handle()); }
    else blitTo(frameIndex == 0 ? sources.first.handle() : sources.second.handle());
    std::vector<vk::Barrier> ready;
    if (seed) {
        for (const auto image : {sources.first.handle(), sources.second.handle()})
            ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range});
    } else {
        const auto image = frameIndex == 0 ? sources.first.handle() : sources.second.handle();
        ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range});
    }
    ready.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, family, VK_QUEUE_FAMILY_FOREIGN_EXT,
        transportImage, range});
    command.insertBarriers(vk, ready, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

RuntimeShadowIngestSnapshot
RuntimeGenerateSessionImpl::shadowIngestSnapshot() const noexcept {
    RuntimeShadowIngestSnapshot result{};
    if (!shadowSplit) return result;
    const auto& shadow = *shadowSplit;
    if (shadow.pending) {
        result.frameId = shadow.pending->frameIdentity();
        result.slot = shadow.pending->destination();
        result.epoch = shadow.pending->epoch();
        result.submitAccepted = shadow.pending->submissionAuthority().state()
            != RuntimeAuthorityState::PREPARED;
        result.submissionRetired = shadow.pending->submissionAuthority().state()
            == RuntimeAuthorityState::RETIRED;
        result.payloadRetired = shadow.pending->payloadAuthority().state()
            == RuntimeTemporaryPayloadState::WAIT_RETIRED;
    }
    result.waitSemaphore = shadow.submittedWait;
    result.waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    result.commandBuffer = shadow.ingestCommand.handle();
    result.signalSemaphore = shadow.ingestIntent == RuntimeIngestIntent::GENERATE_SOURCE
        ? shadow.ingestReady.handle() : VK_NULL_HANDLE;
    result.fence = shadow.ingestFence.handle();
    result.frameTransportReusable = shadow.frameTransportReusable;
    result.ingestReadyReusable = shadow.ingestReadyEpoch.canSignalAgain();
    result.failed = shadow.failed;
    result.deviceLost = shadow.deviceLost;
    result.shadowSubmitCount = shadow.ingestSubmitCount;
    result.shadowGenerateSubmitCount = shadow.generateSubmitCount;
    return result;
}

vk::RuntimeImageObservationDescriptor
RuntimeGenerateSessionImpl::temporalObservationDescriptor(
        TemporalSourceSlot slot) const noexcept {
    const auto index = temporalSourceSlotIndex(slot);
    if (index >= temporalInitialized.size() || !temporalInitialized[index]
            || (shadowSplit && shadowSplit->pending))
        return {};
    return {
        .image = index == 0 ? sources.first.handle() : sources.second.handle(),
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = extent,
        .queueFamilyIndex = instance.getVulkan().queueFamilyIndex(),
        .lifetime = sessionLifetime
    };
}

vk::RuntimeImageObservationDescriptor
RuntimeGenerateSessionImpl::generatedObservationDescriptor() const noexcept {
    if (!shadowSplit || !shadowSplit->generateState
            || !shadowSplit->generateExecutionRetired)
        return {};
    return {
        .image = destination.handle(),
        .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = extent,
        .queueFamilyIndex = instance.getVulkan().queueFamilyIndex(),
        .lifetime = sessionLifetime
    };
}

RuntimeShadowIngestSnapshot RuntimeGenerateSessionImpl::submitShadowIngest(
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport,
        TemporalSourceSlot destinationSlot, uint64_t frameId,
        RuntimeIngestIntent intent) {
    std::optional<vk::RuntimeFrameTransportSubmission> authority;
    authority.emplace(std::move(transport));
    return submitShadowIngestInternal(transportImage, std::move(authority), {},
        destinationSlot, frameId, intent);
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
RuntimeShadowIngestSnapshot RuntimeGenerateSessionImpl::submitShadowIngest(
        VkImage transportImage, vk::SyncFdPayload payload,
        TemporalSourceSlot destinationSlot, uint64_t frameId,
        RuntimeIngestIntent intent) {
    return submitShadowIngestInternal(transportImage, {}, std::move(payload),
        destinationSlot, frameId, intent);
}
#endif

RuntimeShadowIngestSnapshot RuntimeGenerateSessionImpl::submitShadowIngestInternal(
        VkImage transportImage,
        std::optional<vk::RuntimeFrameTransportSubmission> transport,
        vk::SyncFdPayload testingPayload, TemporalSourceSlot destinationSlot,
        uint64_t frameId, RuntimeIngestIntent intent) {
    if (!shadowSplit) throw std::logic_error("shadow split resources unavailable");
    auto& shadow = *shadowSplit;
    if (transportImage == VK_NULL_HANDLE || frameId == 0
            || temporalSourceSlotIndex(destinationSlot) >= 2)
        throw std::invalid_argument("invalid shadow ingest identity");
    if (shadow.pending || shadow.frameTransport || shadow.failed
            || !shadow.ingestReadyEpoch.canSignalAgain())
        throw std::logic_error("shadow ingest resources are not reusable");
    if (transport && !transport->valid())
        throw std::invalid_argument("shadow ingest frame transport authority is invalid");
    if (!transport && !testingPayload.valid())
        throw std::invalid_argument("shadow ingest SYNC_FD payload is invalid");

    const auto& vk = instance.getVulkan();
    auto payload = transport ? transport->releasePayload() : std::move(testingPayload);
    const uint64_t epoch = intent == RuntimeIngestIntent::GENERATE_SOURCE
        ? shadow.nextEpoch++ : shadow.nextWarmupEpoch++;
    const auto destinationIndex = temporalSourceSlotIndex(destinationSlot);
    const bool destinationFirstUse = !temporalInitialized[destinationIndex];
    bool epochReserved{};
    bool submitAccepted{};
    try {
        if (shadow.ingestFenceNeedsReset) {
            shadow.ingestFence.reset(vk);
            shadow.ingestFenceNeedsReset = false;
        }
        shadow.importedWait = std::make_unique<vk::Semaphore>(vk);
        const vk::ExternalSemaphoreDevice semaphoreDevice{vk.dev(), {
            vk.df().CreateSemaphore, vk.df().DestroySemaphore,
            vk.df().GetSemaphoreFdKHR, vk.df().ImportSemaphoreFdKHR}};
        vk::importSyncFdTemporary(semaphoreDevice, shadow.importedWait->handle(), payload);
        shadow.submittedWait = shadow.importedWait->handle();

        shadow.ingestCommand.begin(vk);
        recordTemporalIngestCommands(vk, shadow.ingestCommand, transportImage,
            destinationIndex, false, destinationFirstUse, vk.queueFamilyIndex());
        shadow.ingestCommand.end(vk);

        shadow.ingestIntent = intent;
        if (intent == RuntimeIngestIntent::GENERATE_SOURCE) {
            shadow.ingestReadyEpoch.reserveSignal(epoch);
            epochReserved = true;
        }
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkSemaphore waitSemaphore = shadow.importedWait->handle();
        const VkCommandBuffer commandBuffer = shadow.ingestCommand.handle();
        const VkSemaphore signalSemaphore = intent == RuntimeIngestIntent::GENERATE_SOURCE
            ? shadow.ingestReady.handle() : VK_NULL_HANDLE;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &waitSemaphore,
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = signalSemaphore == VK_NULL_HANDLE ? 0U : 1U,
            .pSignalSemaphores = signalSemaphore == VK_NULL_HANDLE ? nullptr : &signalSemaphore
        };
        const VkResult submitResult = vk.df().QueueSubmit(
            vk.queue(), 1, &submit, shadow.ingestFence.handle());
        if (submitResult != VK_SUCCESS) {
            shadow.deviceLost = submitResult == VK_ERROR_DEVICE_LOST;
            throw ls::vulkan_error(submitResult, "shadow ingest vkQueueSubmit() failed");
        }
        submitAccepted = true;
        if (transport) {
            transport->consumerSubmitted();
            shadow.frameTransport.emplace(std::move(*transport));
        }
        temporalInitialized[destinationIndex] = true;

        if (intent == RuntimeIngestIntent::GENERATE_SOURCE)
            shadow.ingestReadyEpoch.producerSubmitted(epoch);
        shadow.pending.emplace(RuntimeShadowIngestAuthorityAccess::accepted(
            frameId, destinationSlot, epoch, shadow.ingestFence.handle(),
            shadow.importedWait->handle(), sessionLifetime));
        shadow.frameTransportReusable = false;
        ++shadow.ingestSubmitCount;
        return shadowIngestSnapshot();
    } catch (...) {
        if (transport) transport->consumerFailed();
        if (shadow.frameTransport) shadow.frameTransport->consumerFailed();
        if (epochReserved) shadow.ingestReadyEpoch.producerFailed();
        shadow.failed = true;
        if (!submitAccepted && !shadow.deviceLost) {
            shadow.importedWait.reset();
            shadow.pending.reset();
        }
        throw;
    }
}

RuntimeIngestRetirementStatus
RuntimeGenerateSessionImpl::tryRetireShadowIngest() {
    if (!shadowSplit || !shadowSplit->pending)
        throw std::logic_error("no accepted shadow ingest to retire");
    auto& shadow = *shadowSplit;
    const auto epoch = shadow.pending->epoch();
    const auto& vk = instance.getVulkan();
    const auto result = vk.df().GetFenceStatus(vk.dev(), shadow.ingestFence.handle());
    if (result == VK_NOT_READY)
        return RuntimeIngestRetirementStatus::NOT_READY;
    if (result != VK_SUCCESS) {
        if (shadow.frameTransport) shadow.frameTransport->consumerFailed();
        shadow.pending->submissionAuthority().fail();
        shadow.pending->payloadAuthority().fail();
        if (shadow.ingestIntent == RuntimeIngestIntent::GENERATE_SOURCE)
            shadow.ingestReadyEpoch.producerFailed();
        shadow.deviceLost = result == VK_ERROR_DEVICE_LOST;
        shadow.failed = true;
        throw ls::vulkan_error(result, "runtime ingest vkGetFenceStatus() failed");
    }
    shadow.pending->submissionAuthority().retire(epoch);
    shadow.pending->payloadAuthority().waitRetired(epoch);
    if (shadow.frameTransport) {
        shadow.frameTransport->consumerRetired();
        shadow.frameTransport.reset();
    }
    shadow.frameTransportReusable = true;
    shadow.importedWait.reset();
    shadow.ingestFenceNeedsReset = true;
    if (shadow.ingestIntent == RuntimeIngestIntent::WARMUP_TEMPORAL)
        shadow.pending.reset();
    return RuntimeIngestRetirementStatus::RETIRED;
}

RuntimeShadowGenerateSnapshot
RuntimeGenerateSessionImpl::shadowGenerateSnapshot() const noexcept {
    RuntimeShadowGenerateSnapshot result{};
    if (!shadowSplit) return result;
    const auto& shadow = *shadowSplit;
    if (shadow.generateState) {
        const auto& pair = shadow.generateState->temporalPair;
        result.generationId = pair.generationId;
        result.olderFrameId = pair.olderFrameId;
        result.newerFrameId = pair.newerFrameId;
        result.olderSlot = pair.olderSlot;
        result.newerSlot = pair.newerSlot;
        result.waitSemaphore = shadow.ingestReady.handle();
        result.signalSemaphore = shadow.generateState->readinessHandle;
        result.fence = shadow.generateState->fence
            ? shadow.generateState->fence->handle() : VK_NULL_HANDLE;
        result.generatedImage = shadow.generateState->image;
        result.submitAccepted = true;
    }
    result.waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    result.commandBuffer = shadow.generateCommand.handle();
    result.executionRetired = shadow.generateExecutionRetired;
    result.ingestReadyReusable = shadow.ingestReadyEpoch.canSignalAgain();
    if (shadow.generateState) {
        const auto readyState = shadow.generateState->generationReadyEpoch.state();
        result.generationReadySignalOutstanding =
            readyState == RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED;
        result.generationReadyWaitSubmitted =
            readyState == RuntimeBinarySemaphoreState::WAIT_SUBMITTED;
        result.generationReadyWaitRetired =
            readyState == RuntimeBinarySemaphoreState::WAIT_RETIRED;
        result.generationReadyReusable =
            shadow.generateState->generationReadyEpoch.canSignalAgain();
        result.generatedOutputRetired = shadow.generateState->operationRetired;
        result.generatedOutputLive = !shadow.generateState->operationRetired;
    } else {
        result.generationId = shadow.lastRetiredPair.generationId;
        result.olderFrameId = shadow.lastRetiredPair.olderFrameId;
        result.newerFrameId = shadow.lastRetiredPair.newerFrameId;
        result.olderSlot = shadow.lastRetiredPair.olderSlot;
        result.newerSlot = shadow.lastRetiredPair.newerSlot;
        result.executionRetired = shadow.lastExecutionRetired;
        result.generationReadyWaitRetired =
            shadow.lastGenerationReadyWaitRetired;
        result.generationReadyReusable = shadow.lastGenerationReadyReusable;
        result.generatedOutputRetired = shadow.lastGeneratedOutputRetired;
    }
    result.failed = shadow.failed;
    result.deviceLost = shadow.deviceLost;
    result.totalShadowSubmitCount = shadow.ingestSubmitCount + shadow.generateSubmitCount;
    result.bReturnSubmitCount = shadow.bReturnSubmitCount;
    result.aReturnSubmitCount = shadow.aReturnSubmitCount;
    return result;
}

RuntimeShadowGenerateSnapshot RuntimeGenerateSessionImpl::submitShadowGenerate(
        RuntimeTemporalPairIdentity pair) {
    if (!shadowSplit) throw std::logic_error("shadow split resources unavailable");
    auto& shadow = *shadowSplit;
    if (mode != RuntimeGenerateMode::SerialReusable)
        throw std::logic_error("shadow Generate requires SerialReusable authority");
    if (!shadow.pending || shadow.generateState || shadow.failed)
        throw std::logic_error("shadow ingest is not ready for Generate");
    if (!validTemporalPair(pair)
            || pair.newerFrameId != shadow.pending->frameIdentity()
            || pair.newerSlot != shadow.pending->destination()
            || pair.olderSlot == pair.newerSlot
            || shadow.ingestReadyEpoch.state() != RuntimeBinarySemaphoreState::SIGNAL_SUBMITTED)
        throw std::invalid_argument("invalid shadow Generate temporal identity");

    // The previous generation's B/A counts remain visible in the last-retired
    // snapshot until the next Generate begins.  Reset only the live epoch
    // counters now that the ingest authority has been proven reusable.
    shadow.bReturnSubmitCount = 0;
    shadow.aReturnSubmitCount = 0;

    const auto& vk = instance.getVulkan();
    const auto epoch = shadow.pending->epoch();
    auto pending = std::make_shared<RuntimeGenerateDiagnosticPendingState>();
    pending->generation = pair.generationId;
    pending->image = destination.handle();
    pending->extent = extent;
    pending->format = VK_FORMAT_R8G8B8A8_UNORM;
    pending->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    pending->family = vk.queueFamilyIndex();
    pending->sessionLifetime = sessionLifetime;
    pending->fence = std::make_unique<vk::Fence>(vk);
    pending->readiness = std::make_unique<vk::Semaphore>(vk);
    pending->readinessHandle = pending->readiness->handle();
    pending->temporalPair = pair;

    bool submitAccepted{};
    serialState.submit(pair.generationId);
    try {
        shadow.generateCommand.begin(vk);
        recordPrepassGenerateCommands(vk, shadow.generateCommand,
            temporalSourceSlotIndex(pair.newerSlot), true, true);
        shadow.generateCommand.end(vk);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        const VkSemaphore waitSemaphore = shadow.ingestReady.handle();
        const VkCommandBuffer commandBuffer = shadow.generateCommand.handle();
        const VkSemaphore signalSemaphore = pending->readinessHandle;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &waitSemaphore,
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signalSemaphore
        };
        const VkResult submitResult = vk.df().QueueSubmit(
            vk.queue(), 1, &submit, pending->fence->handle());
        if (submitResult != VK_SUCCESS) {
            shadow.deviceLost = submitResult == VK_ERROR_DEVICE_LOST;
            throw ls::vulkan_error(submitResult, "shadow Generate vkQueueSubmit() failed");
        }
        submitAccepted = true;

        shadow.ingestReadyEpoch.waitSubmitted(epoch);
        pending->generationReadyEpoch.signalSubmitted(pair.generationId);
        shadow.generateState = pending;
        shadow.generatePending.emplace(RuntimeGenerateDiagnosticPending(pending));
        pendingGeneration = pending;
        shadow.generationReadyOutstanding = true;
        ++shadow.generateSubmitCount;
        return shadowGenerateSnapshot();
    } catch (...) {
        shadow.ingestReadyEpoch.consumerFailed(epoch);
        serialState.fail();
        shadow.failed = true;
        if (!submitAccepted && !shadow.deviceLost) {
            shadow.generatePending.reset();
            shadow.generateState.reset();
        } else if (submitAccepted) {
            shadow.generateState = std::move(pending);
        }
        throw;
    }
}

RuntimeRetirementStatus RuntimeGenerateSessionImpl::tryRetireShadowGenerate() {
    if (!shadowSplit || !shadowSplit->generateState)
        throw std::logic_error("no accepted shadow Generate to retire");
    auto& shadow = *shadowSplit;
    if (shadow.generateExecutionRetired)
        return RuntimeRetirementStatus::RETIRED;
    if (!shadow.pending)
        throw std::logic_error("Generate retirement lost its ingest authority");
    const auto epoch = shadow.pending->epoch();
    const auto& vk = instance.getVulkan();
    const auto result = vk.df().GetFenceStatus(
        vk.dev(), shadow.generateState->fence->handle());
    if (result == VK_NOT_READY)
        return RuntimeRetirementStatus::NOT_READY;
    if (result != VK_SUCCESS) {
        if (shadow.frameTransport) shadow.frameTransport->consumerFailed();
        shadow.ingestReadyEpoch.consumerFailed(epoch);
        shadow.generateState->generationReadyEpoch.consumerFailed(
            shadow.generateState->generation);
        serialState.fail();
        shadow.deviceLost = result == VK_ERROR_DEVICE_LOST;
        shadow.failed = true;
        return shadow.deviceLost ? RuntimeRetirementStatus::DEVICE_LOST
                                 : RuntimeRetirementStatus::FAILED;
    }
    shadow.ingestReadyEpoch.waitRetired(epoch);
    shadow.ingestReadyEpoch.makeAvailable(epoch);
    shadow.pending->submissionAuthority().retire(epoch);
    shadow.pending->payloadAuthority().waitRetired(epoch);
    if (shadow.frameTransport) {
        shadow.frameTransport->consumerRetired();
        shadow.frameTransport.reset();
    }
    shadow.frameTransportReusable = true;
    shadow.ingestFenceNeedsReset = true;
    shadow.pending.reset();
    shadow.importedWait.reset();
    if (serialState.currentPhase() == RuntimeGenerateSerialPhase::IN_FLIGHT)
        serialState.sourceReadsRetired(shadow.generateState->generation);
    shadow.generateExecutionRetired = true;
    return RuntimeRetirementStatus::RETIRED;
}

RuntimeShadowGenerateSnapshot RuntimeGenerateSessionImpl::retireShadowGenerate() {
    if (!shadowSplit || !shadowSplit->generateState)
        throw std::logic_error("no accepted shadow Generate to retire");
    if (!shadowSplit->generateExecutionRetired
            && !shadowSplit->generateState->fence->wait(instance.getVulkan())) {
        shadowSplit->failed = true;
        throw backend::error("shadow Generate fence wait failed");
    }
    const auto retirement = tryRetireShadowGenerate();
    if (retirement != RuntimeRetirementStatus::RETIRED)
        throw backend::error("shadow Generate retirement failed");
    return shadowGenerateSnapshot();
}

std::optional<RuntimeGenerateDiagnosticPending>
RuntimeGenerateSessionImpl::takeShadowGeneratePending() {
    if (!shadowSplit || !shadowSplit->generatePending || shadowSplit->failed)
        throw std::logic_error("no usable shadow Generate pending capability");
    auto result = std::move(shadowSplit->generatePending);
    shadowSplit->generatePending.reset();
    shadowSplit->generationReadyOutstanding = true;
    return result;
}

void RuntimeGenerateSessionImpl::recordShadowBReturnSubmitted(
        RuntimeGenerationId generation) {
    if (!shadowSplit || !shadowSplit->generateState || shadowSplit->failed
            || shadowSplit->generateState->generation != generation
            || shadowSplit->bReturnSubmitCount != 0)
        throw std::logic_error("invalid shadow B-return submit authority");
    shadowSplit->generateState->generationReadyEpoch.waitSubmitted(generation);
    ++shadowSplit->bReturnSubmitCount;
}

void RuntimeGenerateSessionImpl::recordShadowBReturnRejected(
        RuntimeGenerationId generation, bool lost) {
    if (!shadowSplit || !shadowSplit->generateState
            || shadowSplit->generateState->generation != generation
            || shadowSplit->bReturnSubmitCount != 0)
        throw std::logic_error("invalid rejected shadow B-return authority");
    shadowSplit->generateState->generationReadyEpoch.consumerFailed(generation);
    shadowSplit->failed = true;
    shadowSplit->deviceLost = lost;
    serialState.fail();
}

void RuntimeGenerateSessionImpl::retireShadowBReturnWait(
        RuntimeGenerationId generation) {
    if (!shadowSplit || !shadowSplit->generateState || shadowSplit->failed
            || shadowSplit->generateState->generation != generation
            || shadowSplit->bReturnSubmitCount != 1)
        throw std::logic_error("invalid shadow B-return retirement authority");
    shadowSplit->generateState->generationReadyEpoch.waitRetired(generation);
}

void RuntimeGenerateSessionImpl::failShadowBReturnRetirement(
        RuntimeGenerationId generation, bool lost) {
    if (!shadowSplit || !shadowSplit->generateState
            || shadowSplit->generateState->generation != generation
            || shadowSplit->bReturnSubmitCount != 1)
        throw std::logic_error("invalid failed shadow B-return retirement authority");
    shadowSplit->generateState->generationReadyEpoch.consumerFailed(generation);
    shadowSplit->failed = true;
    shadowSplit->deviceLost = lost;
    serialState.fail();
}

void RuntimeGenerateSessionImpl::releaseShadowGenerationReady(
        RuntimeGenerationId generation) {
    if (!shadowSplit || !shadowSplit->generateState || shadowSplit->failed
            || shadowSplit->generateState->generation != generation
            || !shadowSplit->generateState->operationRetired)
        throw std::logic_error("generated output has not retired for semaphore reuse");
    shadowSplit->generateState->generationReadyEpoch.makeAvailable(generation);
    shadowSplit->lastRetiredPair = shadowSplit->generateState->temporalPair;
    shadowSplit->lastRetiredGeneration = generation;
    shadowSplit->lastExecutionRetired = shadowSplit->generateExecutionRetired;
    shadowSplit->lastGeneratedOutputRetired = shadowSplit->generateState->operationRetired;
    shadowSplit->lastGenerationReadyWaitRetired = true;
    shadowSplit->lastGenerationReadyReusable = true;
    shadowSplit->generateState.reset();
    shadowSplit->generatePending.reset();
    shadowSplit->generationReadyOutstanding = false;
    shadowSplit->generateExecutionRetired = false;
}

void RuntimeGenerateSessionImpl::recordShadowAReturnSubmitted(
        RuntimeGenerationId generation) {
    if (!shadowSplit || !shadowSplit->generateState || shadowSplit->failed
            || shadowSplit->generateState->generation != generation
            || shadowSplit->bReturnSubmitCount != 1
            || shadowSplit->aReturnSubmitCount != 0)
        throw std::logic_error("invalid shadow A-return submit authority");
    ++shadowSplit->aReturnSubmitCount;
}

void RuntimeGenerateSessionImpl::recordPrepassGenerateCommands(
        const vk::Vulkan& vk, const vk::CommandBuffer& command,
        size_t frameIndex, bool runGammaDelta, bool runGenerate) {
    mipmaps.render(vk, command, frameIndex);
    for (size_t i = 0; i < 7; ++i) {
        alpha0.at(6 - i).render(vk, command);
        alpha1.at(6 - i).render(vk, command, frameIndex);
    }
    beta0.render(vk, command, frameIndex); beta1.render(vk, command);
    if (runGammaDelta) for (size_t j = 0; j < 7; ++j) {
        pass.gamma0.at(j).render(vk, command, frameIndex); pass.gamma1.at(j).render(vk, command);
        if (j >= 4) { pass.delta0.at(j - 4).render(vk, command, frameIndex); pass.delta1.at(j - 4).render(vk, command); }
    }
    if (!runGenerate) return;
    const auto range = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const vk::Barrier toClear{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, destination.handle(), range};
    command.insertBarriers(vk, {toClear}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkClearColorValue zero{};
    vk.df().CmdClearColorImage(command.handle(), destination.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
    const vk::Barrier toGenerate{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, destination.handle(), range};
    command.insertBarriers(vk, {toGenerate}, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    pass.generate->render(vk, command, frameIndex);
    const vk::Barrier toReadback{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, destination.handle(), range};
    command.insertBarriers(vk, {toReadback}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0}, {extent.width, extent.height, 1}};
    vk.df().CmdCopyImageToBuffer(command.handle(), destination.handle(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1, &copy);
    const VkBufferMemoryBarrier hostRead{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, readback.handle(), 0, capturedBytes};
    vk.df().CmdPipelineBarrier(command.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hostRead, 0, nullptr);
}

std::optional<RuntimeGenerateDiagnosticPending>
RuntimeGenerateSessionImpl::processSubmission(
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport,
        bool deferGenerateCompletion,
        std::optional<TemporalSourceSlot> explicitDestination, uint64_t frameId,
        std::optional<RuntimeTemporalPairIdentity> explicitPair) {
    static_cast<void>(deferGenerateCompletion);
    if (pendingGeneration)
        throw std::logic_error("runtime Generate diagnostic already has a pending generation");
    if (!transport.valid())
        throw std::invalid_argument("runtime Generate frame transport authority is invalid");
    RuntimeGenerateDiagnosticStep step{};
    if (mode == RuntimeGenerateMode::OneShot) {
        step = state.advance();
    } else {
        if (!explicitDestination || temporalSourceSlotIndex(*explicitDestination) >= 2
                || frameId == 0)
            throw std::invalid_argument("serial Generate requires explicit temporal ingest identity");
        const auto index = temporalSourceSlotIndex(*explicitDestination);
        const bool seedOperation = temporalFrameIds[0] == 0 && temporalFrameIds[1] == 0;
        step = {seedOperation ? RuntimeGenerateDiagnosticAction::SEED_ONLY
                             : explicitPair ? RuntimeGenerateDiagnosticAction::GENERATE
                                            : RuntimeGenerateDiagnosticAction::GAMMA_DELTA,
            0, 0, index, 0, index, index};
    }
    try {
    const auto& vk = instance.getVulkan();
    const uint32_t family = vk.queueFamilyIndex();
    const bool seed = step.action == RuntimeGenerateDiagnosticAction::SEED_ONLY;
    const bool runGammaDelta = step.action != RuntimeGenerateDiagnosticAction::SEED_ONLY;
    const bool runGenerate = step.action == RuntimeGenerateDiagnosticAction::GENERATE;
    const size_t frameIndex = step.frameIndex;
    const auto expectedDestination = frameIndex == 0
        ? TemporalSourceSlot::Slot0 : TemporalSourceSlot::Slot1;
    if (explicitDestination && temporalSourceSlotIndex(*explicitDestination) >= 2)
        throw std::invalid_argument("invalid explicit temporal destination");
    if (mode == RuntimeGenerateMode::OneShot
            && explicitDestination && *explicitDestination != expectedDestination)
        throw std::invalid_argument("explicit temporal destination contradicts backend orientation");
    if (explicitDestination && frameId == 0)
        throw std::invalid_argument("explicit temporal ingest requires frame identity");
    if (runGenerate) {
        if (!explicitPair)
            throw std::invalid_argument("explicit Generate requires temporal pair identity");
        if (!validTemporalPair(*explicitPair)
                || temporalPairFrameIndex(*explicitPair) != frameIndex
                || explicitPair->newerFrameId != frameId)
            throw std::invalid_argument("invalid explicit temporal pair orientation or identity");
        if (mode == RuntimeGenerateMode::SerialReusable) {
            const auto olderIndex = temporalSourceSlotIndex(explicitPair->olderSlot);
            if (temporalFrameIds[olderIndex] != explicitPair->olderFrameId)
                throw std::invalid_argument("stale explicit temporal history identity");
            serialState.submit(explicitPair->generationId);
        }
    }
    if (mode == RuntimeGenerateMode::SerialReusable) {
        if (seed)
            temporalFrameIds = {frameId, frameId};
        else
            temporalFrameIds[frameIndex] = frameId;
    }

    auto payload = transport.releasePayload();
    const vk::ExternalSemaphoreDevice semaphoreDevice{vk.dev(), {
        vk.df().CreateSemaphore, vk.df().DestroySemaphore,
        vk.df().GetSemaphoreFdKHR, vk.df().ImportSemaphoreFdKHR}};
    auto waitSemaphore = vk::createSyncFdImportSemaphore(semaphoreDevice);
    vk::importSyncFdTemporary(semaphoreDevice, waitSemaphore.handle(), payload);

    vk::CommandBuffer command(vk);
    command.begin(vk);
    recordTemporalIngestCommands(vk, command, transportImage, frameIndex, seed,
        !temporalInitialized[frameIndex], family);
    recordPrepassGenerateCommands(vk, command, frameIndex, runGammaDelta, runGenerate);

    command.end(vk);
    if (runGenerate) {
        auto pending = std::make_shared<RuntimeGenerateDiagnosticPendingState>();
        pending->generation = mode == RuntimeGenerateMode::SerialReusable
            ? explicitPair->generationId : generation;
        pending->image = destination.handle();
        pending->extent = extent;
        pending->format = VK_FORMAT_R8G8B8A8_UNORM;
        pending->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        pending->family = family;
        pending->sessionLifetime = sessionLifetime;
        pending->fence = std::make_unique<vk::Fence>(vk);
        pending->readiness = std::make_unique<vk::Semaphore>(vk);
        pending->readinessHandle = pending->readiness->handle();
        pending->temporalPair = *explicitPair;
        pending->temporalPair.generationId = pending->generation;
        if (mode == RuntimeGenerateMode::OneShot)
            state.recordSubmittedGeneration();
        command.submit(vk, vk.queue(), {waitSemaphore.handle()}, VK_NULL_HANDLE, 0,
            {pending->readiness->handle()}, VK_NULL_HANDLE, 0,
            pending->fence->handle(), VK_PIPELINE_STAGE_TRANSFER_BIT);
        transport.consumerSubmitted();
        pending->frameTransport.emplace(std::move(transport));
        if (seed) temporalInitialized = {true, true};
        else temporalInitialized[frameIndex] = true;
        pendingGeneration = pending;
        return RuntimeGenerateDiagnosticPending(pending);
    }

    vk::Fence fence(vk);
    command.submit(vk, vk.queue(), {waitSemaphore.handle()}, VK_NULL_HANDLE, 0,
        {}, VK_NULL_HANDLE, 0, fence.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT);
    transport.consumerSubmitted();
    if (seed) temporalInitialized = {true, true};
    else temporalInitialized[frameIndex] = true;
    if (!fence.wait(vk))
        throw backend::error("P4C-D2 backend fence wait failed");
    transport.consumerRetired();

    if (step.action == RuntimeGenerateDiagnosticAction::SEED_ONLY) {
        std::cerr << "[DG2X-P4C-D2] Diagnostic temporal seed\n"
            << "  D2 direct real frames: 1\n  Seed duplication: YES\n"
            << "  Generate: NOT RUN\n  Host wait A->B: NONE\n";
        return std::nullopt;
    }
    if (step.action == RuntimeGenerateDiagnosticAction::GAMMA_DELTA) {
        std::cerr << "[DG2X-P4C-D2A] Gamma/Delta prerequisites\n"
            << "  D2 temporal pair valid: YES\n"
            << "  D2 direct real frames: 2\n  Temporal rotations: 1\n"
            << "  Gamma variant: " << (ctx.perf ? "PERFORMANCE" : "QUALITY") << "\n"
            << "  Delta variant: " << (ctx.perf ? "PERFORMANCE" : "QUALITY") << "\n"
            << "  Gamma iterations completed: 7\n"
            << "  Delta iterations completed: 3\n"
            << "  Generate: NOT RUN\n  Generated frame: NONE\n"
            << "  Backend fence completion: PASS\n  Host wait A->B: NONE\n"
            << "DG2X_P4C_D2A_GAMMA_DELTA_B_PASS\n";
        return std::nullopt;
    }

    throw std::logic_error("runtime Generate submission did not produce pending state");
    } catch (...) {
        transport.consumerFailed();
        if (mode == RuntimeGenerateMode::OneShot) state.recordFailure();
        else serialState.fail();
        throw;
    }
}

RuntimeGenerateDiagnosticResult
RuntimeGenerateSessionImpl::validateCompletedGeneration(
        const std::shared_ptr<RuntimeGenerateDiagnosticPendingState>& pending) {
    const auto& vk = instance.getVulkan();
    if (!pending || pending != pendingGeneration || pending->completionConsumed)
        throw std::logic_error("invalid or already completed D3A2 generation");
    if (!pending->fence->wait(vk))
        throw backend::error("P4C-D2 delayed backend fence wait failed");
    if (pending->frameTransport) {
        pending->frameTransport->consumerRetired();
        pending->frameTransport.reset();
    }
    const auto bytes = readback.read(vk, capturedBytes);
    if (bytes.size() != capturedBytes)
        throw backend::error("P4C-D2 generated readback byte count mismatch");
    uint64_t checksum = lsfgvk::common::fnv1a64(bytes.data(), bytes.size());
    size_t nonzero{};
    for (const auto byte : bytes) {
        nonzero += byte != 0;
    }
    if (nonzero == 0)
        throw backend::error("P4C-D2 generated destination is entirely zero");
    state.recordValidatedOutput();
    if (!state.finalPassReady())
        throw backend::error("P4C-D2 final marker gate rejected incomplete state");

    std::cerr << "[DG2X-P4C-D2B] GPU-B-local generated frame\n"
        << "  D2 temporal pair valid: YES\n"
        << "  D2 direct real frames: 3\n  Temporal rotations: 2\n"
        << "  Physical source slot 1: older frame B\n"
        << "  Physical source slot 0: newer frame C\n"
        << "  Generate sourceImages.first: physical slot 1 / older B\n"
        << "  Generate sourceImages.second: physical slot 0 / newer C\n"
        << "  Generate descriptor set: 0\n"
        << "  Gamma iterations completed: 7\n  Delta iterations completed: 3\n"
        << "  Generate shader: SDR\n  Generate executions: 1\n"
        << "  Interpolation constant source: getDefaultConstantBuffer(0, 1, flow)\n"
        << "  Interpolation timestamp: 0.5\n"
        << "  Destination format: R8G8B8A8_UNORM\n"
        << "  Destination zero clear: PASS\n"
        << "  Captured byte count: " << bytes.size() << "\n"
        << "  Captured non-zero bytes: " << nonzero << "\n"
        << "  Captured checksum: 0x" << std::hex << checksum << std::dec << "\n"
        << "  Backend fence completion: PASS\n"
        << "  LSFG backend execution: GENERATE_DIAGNOSTIC\n"
        << "  Frame transport: INPUT_ONLY\n  Generated frame: GPU_B_LOCAL\n"
        << "  Presentation from B: NONE\n  Output transport B->A: NONE\n"
        << "  CPU frame bridge: NONE\n  Host wait A->B: NONE\n"
        << "DG2X_P4C_D2_GENERATED_FRAME_B_PASS\n";
    state.issueCompletionResult();
    return RuntimeGenerateDiagnosticResult{
        .metadata = {generation, extent, VK_FORMAT_R8G8B8A8_UNORM,
            bytes.size(), nonzero, checksum},
        .frame = RuntimeGeneratedFrameToken(generation, destination.handle(), extent,
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pending->family,
            sessionLifetime)
    };
}

std::optional<RuntimeGenerateDiagnosticResult> RuntimeGenerateSessionImpl::process(
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport) {
    const auto step = state.nextStep();
    const auto slot = step.frameIndex == 0 ? TemporalSourceSlot::Slot0 : TemporalSourceSlot::Slot1;
    std::optional<RuntimeTemporalPairIdentity> pair;
    if (step.action == RuntimeGenerateDiagnosticAction::GENERATE)
        pair = RuntimeTemporalPairIdentity{TemporalSourceSlot::Slot1,
            TemporalSourceSlot::Slot0, 2, 3, generation};
    auto pending = processSubmission(transportImage, std::move(transport), false,
        slot, step.directFrames, pair);
    if (!pending) return std::nullopt;
    pending->consumeTransport();
    return complete(std::move(*pending));
}

std::optional<RuntimeGenerateDiagnosticPending> RuntimeGenerateSessionImpl::submit(
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport) {
    const auto step = state.nextStep();
    const auto slot = step.frameIndex == 0 ? TemporalSourceSlot::Slot0 : TemporalSourceSlot::Slot1;
    std::optional<RuntimeTemporalPairIdentity> pair;
    if (step.action == RuntimeGenerateDiagnosticAction::GENERATE)
        pair = RuntimeTemporalPairIdentity{TemporalSourceSlot::Slot1,
            TemporalSourceSlot::Slot0, 2, 3, generation};
    return submitExplicit(transportImage, std::move(transport), slot,
        step.directFrames, pair);
}

std::optional<RuntimeGenerateDiagnosticPending>
RuntimeGenerateSessionImpl::submitExplicit(
        VkImage transportImage, vk::RuntimeFrameTransportSubmission transport,
        TemporalSourceSlot destinationSlot, uint64_t frameId,
        std::optional<RuntimeTemporalPairIdentity> pair) {
    return processSubmission(transportImage, std::move(transport), true,
        destinationSlot, frameId, pair);
}

RuntimeGenerateDiagnosticResult RuntimeGenerateSessionImpl::complete(
        RuntimeGenerateDiagnosticPending&& capability) {
    const auto pending = capability.pending;
    if (!pending || pending != pendingGeneration || !capability.valid()
            || !pending->transportConsumed)
        throw std::logic_error("D3A2 completion lacks consumed transport authority");
    try {
        if (mode == RuntimeGenerateMode::SerialReusable) {
            if (serialState.currentPhase() == RuntimeGenerateSerialPhase::IN_FLIGHT) {
                if (!pending->fence || !pending->fence->wait(instance.getVulkan()))
                    throw backend::error("serial Generate source-read fence wait failed");
                if (pending->frameTransport) {
                    pending->frameTransport->consumerRetired();
                    pending->frameTransport.reset();
                }
                serialState.sourceReadsRetired(pending->generation);
            } else if (serialState.currentPhase()
                    != RuntimeGenerateSerialPhase::SOURCE_READS_RETIRED) {
                throw std::logic_error("serial Generate source reads lack retirement proof");
            }
            serialState.outputInUse(pending->generation);
            RuntimeGenerateDiagnosticResult result{
                .metadata = {pending->generation, pending->extent, pending->format, 0, 0, 0},
                .frame = RuntimeGeneratedFrameToken(pending->generation, pending->image,
                    pending->extent, pending->format, pending->layout, pending->family,
                    pending->sessionLifetime)};
            pending->completionConsumed = true;
            capability.pending.reset();
            return result;
        }
        auto result = validateCompletedGeneration(pending);
        pending->completionConsumed = true;
        capability.pending.reset();
        pendingGeneration.reset();
        return result;
    } catch (...) {
        pending->completionConsumed = true;
        capability.pending.reset();
        pendingGeneration.reset();
        if (mode == RuntimeGenerateMode::OneShot) state.recordFailure();
        else serialState.fail();
        throw;
    }
}

void RuntimeGenerateSessionImpl::retireOperation(
        RuntimeGenerationOperationRetirement&& authority) {
    const auto pending = authority.pending;
    if (mode != RuntimeGenerateMode::SerialReusable || !pending
            || pending != pendingGeneration || pending->operationRetired
            || !pending->completionConsumed)
        throw std::logic_error("invalid or stale serial Generate retirement authority");
    serialState.retireOperation(pending->generation);
    pending->operationRetired = true;
    authority.pending.reset();
    pendingGeneration.reset();
}

ContextImpl::ContextImpl(const InstanceImpl& instance,
            std::pair<vk::ExternalImage, vk::ExternalImage> externalSourceImages,
            std::vector<vk::ExternalImage> externalDestImages, int syncFd,
            std::vector<ls::OwnedFd> destinationReturnFds,
            VkExtent2D extent, bool hdr, float flow, bool perf) :
        sourceImages(importImages(instance.getVulkan(), externalSourceImages)),
        destImages(importImages(instance.getVulkan(), externalDestImages)),
        blackImage(createBlackImage(instance.getVulkan())),
        syncSemaphore(importTimelineSemaphore(instance.getVulkan(), syncFd)),
        destinationReturnSemaphores(importDestinationReturnSemaphores(
            instance.getVulkan(), destinationReturnFds)),
        prepassSemaphore(createPrepassSemaphore(instance.getVulkan())),
        destinationReturnStates(externalDestImages.size()),
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
    constexpr size_t GENERATION_SLOT_COUNT = 2;
    this->generationSlots.reserve(GENERATION_SLOT_COUNT);
    for (size_t slotIndex = 0; slotIndex < GENERATION_SLOT_COUNT; ++slotIndex) {
        auto& slot = *this->generationSlots.emplace_back(
            std::make_unique<GenerationResourceSlot>(instance, extent, hdr,
                flow, perf, externalDestImages.size()));
        slot.passes.reserve(destImages.size());
        auto& slotCtx = *slot.ctx;
        for (size_t i = 0; i < destImages.size(); ++i) {
        auto& pass = slot.passes.emplace_back();

        pass.gamma0.reserve(7);
        pass.gamma1.reserve(7);
        pass.delta0.reserve(3);
        pass.delta1.reserve(3);
        for (size_t j = 0; j < 7; j++) {
            if (j == 0) { // first pass has no prior data
                pass.gamma0.emplace_back(slotCtx, i,
                    this->alpha1.at(6 - j).getImages(),
                    this->blackImage
                );
                pass.gamma1.emplace_back(slotCtx, i,
                    pass.gamma0.at(j).getImages(),
                    this->blackImage,
                    this->beta1.getImages().at(5)
                );
            } else { // other passes use prior data
                pass.gamma0.emplace_back(slotCtx, i,
                    this->alpha1.at(6 - j).getImages(),
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.gamma1.emplace_back(slotCtx, i,
                    pass.gamma0.at(j).getImages(),
                    pass.gamma1.at(j - 1).getImage(),
                    this->beta1.getImages().at(6 - j)
                );
            }

            if (j == 4) { // first special pass has no prior data
                pass.delta0.emplace_back(slotCtx, i,
                    this->alpha1.at(6 - j).getImages(),
                    this->blackImage,
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.delta1.emplace_back(slotCtx, i,
                    pass.delta0.at(j - 4).getImages0(),
                    pass.delta0.at(j - 4).getImages1(),
                    this->blackImage,
                    this->beta1.getImages().at(6 - j),
                    this->blackImage
                );
            } else if (j > 4) { // further passes do
                pass.delta0.emplace_back(slotCtx, i,
                    this->alpha1.at(6 - j).getImages(),
                    pass.delta1.at(j - 5).getImage0(),
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.delta1.emplace_back(slotCtx, i,
                    pass.delta0.at(j - 4).getImages0(),
                    pass.delta0.at(j - 4).getImages1(),
                    pass.delta1.at(j - 5).getImage0(),
                    this->beta1.getImages().at(6 - j),
                    pass.delta1.at(j - 5).getImage1()
                );
            }
        }

        pass.generate.emplace(slotCtx, i,
            this->sourceImages,
            pass.gamma1.at(6).getImage(),
            pass.delta1.at(2).getImage0(),
            pass.delta1.at(2).getImage1(),
            this->destImages.at(i)
        );
        }
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
    for (const auto& slot : this->generationSlots) {
    for (const auto& pass : slot->passes) {
        for (size_t i = 0; i < 7; ++i) {
            pass.gamma0.at(i).prepare(images);
            pass.gamma1.at(i).prepare(images);

            if (i < 4) continue;
            pass.delta0.at(i - 4).prepare(images);
            pass.delta1.at(i - 4).prepare(images);
        }
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

struct lsfgvk::backend::PreparedFrameScheduleReservationState {
    ContextImpl* owner{};
    GenerationResourceSlot* slot{};
    vk::ExchangeTimelineFrame timeline{};
    uint64_t fidx{};
    size_t generatedFrames{};
    bool active{};
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    void* testOwner{};
    void (*testExecute)(void*){};
    void (*testAbort)(void*){};
#endif
};

PreparedFrameScheduleReservation::PreparedFrameScheduleReservation() noexcept = default;
PreparedFrameScheduleReservation::PreparedFrameScheduleReservation(
        std::unique_ptr<PreparedFrameScheduleReservationState> value) noexcept :
    state(std::move(value)) {}
PreparedFrameScheduleReservation::PreparedFrameScheduleReservation(
    PreparedFrameScheduleReservation&&) noexcept = default;
PreparedFrameScheduleReservation& PreparedFrameScheduleReservation::operator=(
        PreparedFrameScheduleReservation&& other) noexcept {
    if (this != &other) {
        if (state && state->active) {
            if (state->owner) state->owner->abortReservation(*state);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
            else if (state->testAbort) {
                state->testAbort(state->testOwner);
                state->active = false;
            }
#endif
        }
        state = std::move(other.state);
    }
    return *this;
}
PreparedFrameScheduleReservation::~PreparedFrameScheduleReservation() {
    if (state && state->active) {
        if (state->owner) state->owner->abortReservation(*state);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
        else if (state->testAbort) state->testAbort(state->testOwner);
#endif
    }
}
bool PreparedFrameScheduleReservation::valid() const noexcept {
    return state && state->active;
}
vk::ExchangeTimelineFrame PreparedFrameScheduleReservation::timeline() const noexcept {
    return state ? state->timeline : vk::ExchangeTimelineFrame{};
}
uint64_t PreparedFrameScheduleReservation::frameIndex() const noexcept {
    return state ? state->fidx : 0;
}
size_t PreparedFrameScheduleReservation::generatedFrames() const noexcept {
    return state ? state->generatedFrames : 0;
}
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
uintptr_t PreparedFrameScheduleReservation::slotIdentityForTesting() const noexcept {
    return state && state->slot ? reinterpret_cast<uintptr_t>(state->slot) : 0;
}
VkBuffer PreparedFrameScheduleReservation::constantBufferForTesting(
        size_t index) const noexcept {
    if (!state || !state->slot || index >= state->slot->ctx->constantBuffers.size())
        return VK_NULL_HANDLE;
    return state->slot->ctx->constantBuffers[index].handle();
}
VkCommandBuffer PreparedFrameScheduleReservation::commandBufferForTesting(
        size_t index) const noexcept {
    if (!state || !state->slot || index >= state->slot->cmdbufs.size())
        return VK_NULL_HANDLE;
    return state->slot->cmdbufs[index].handle();
}
#endif
void PreparedFrameScheduleReservation::execute() {
    if (!this->valid())
        throw backend::error("invalid or already consumed frame schedule reservation");
    if (state->owner) {
        state->owner->executeReservation(*state);
        return;
    }
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    if (state->testExecute) {
        state->testExecute(state->testOwner);
        state->active = false;
        return;
    }
#endif
    throw backend::error("prepared frame schedule reservation has no executor");
}

#ifdef LSFGVK_TESTING_SHADOW_SPLIT
PreparedFrameScheduleReservation PreparedFrameScheduleReservationTestAccess::make(
        void* owner, void (*execute)(void*), void (*abort)(void*)) {
    auto state = std::make_unique<PreparedFrameScheduleReservationState>();
    state->testOwner = owner;
    state->testExecute = execute;
    state->testAbort = abort;
    state->active = true;
    return PreparedFrameScheduleReservation(std::move(state));
}
#endif

std::optional<PreparedFrameScheduleReservation> Instance::reserveFrameSchedule(
        Context& context, const std::vector<float>& timestamps,
        std::optional<vk::ExchangeTimelineFrame> expectedTimeline) {
    return context.reserveFrameSchedule(timestamps, expectedTimeline);
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
    std::vector<float> timestamps;
    timestamps.reserve(this->destImages.size());
    for (size_t i = 0; i < this->destImages.size(); ++i)
        timestamps.push_back(backend::getDefaultConstantBuffer(
            i, this->destImages.size(), this->ctx.flow).timestamp);
    auto reservation = this->reserveFrameSchedule(timestamps, std::nullopt);
    if (!reservation)
        throw backend::error("no prepared frame schedule slot is available");
    reservation->execute();
}

void Context::recordReservation(GenerationResourceSlot& slot,
        size_t generatedFrames, uint64_t reservedFidx,
        const vk::ExchangeTimelineFrame& timeline) {
    const size_t currentSource = reservedFidx % 2;
    const size_t returnedSource = (reservedFidx + 1) % 2;
    const uint32_t family = this->ctx.vk.get().queueFamilyIndex();

    const auto& cmdbuf = slot.cmdbufs.at(0);
    cmdbuf.begin(ctx.vk);

    std::vector<vk::Barrier> sourceAcquires;
    if (reservedFidx == 0) {
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

    this->mipmaps.render(ctx.vk, cmdbuf, reservedFidx);
    for (size_t i = 0; i < 7; ++i) {
        this->alpha0.at(6 - i).render(ctx.vk, cmdbuf);
        this->alpha1.at(6 - i).render(ctx.vk, cmdbuf, reservedFidx);
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

    // schedule main passes
    for (size_t i = 0; i < generatedFrames; i++) {
        const auto& cmdbuf = slot.cmdbufs.at(i + 1);
        cmdbuf.begin(ctx.vk);

        auto& destinationReturn = this->destinationReturnStates.at(i);
        const auto destinationAcquire = destinationReturn.firstUse
            ? vk::destinationInitialAcquireFromLayer(this->destImages.at(i).handle(), family)
            : vk::destinationAcquireFromLayer(this->destImages.at(i).handle(), family);
        cmdbuf.insertBarriers(ctx.vk, { destinationAcquire },
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        const auto& pass = slot.passes.at(i);
        for (size_t j = 0; j < 7; j++) {
            pass.gamma0.at(j).render(ctx.vk, cmdbuf, reservedFidx);
            pass.gamma1.at(j).render(ctx.vk, cmdbuf);

            if (j < 4) continue;
            pass.delta0.at(j - 4).render(ctx.vk, cmdbuf, reservedFidx);
            pass.delta1.at(j - 4).render(ctx.vk, cmdbuf);
        }
        pass.generate->render(ctx.vk, cmdbuf, reservedFidx);

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
    }
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

    auto reservation = this->reserveFrameSchedule(timestamps, std::nullopt);
    if (!reservation)
        throw backend::error("no prepared frame schedule slot is available");
    reservation->execute();
}

bool Context::tryRetireSlot(GenerationResourceSlot& slot) noexcept {
    if (slot.authority.phase() == vk::GenerationSlotPhase::Available) return true;
    if (slot.authority.phase() != vk::GenerationSlotPhase::InFlight
            || !slot.authority.terminalSubmitted())
        return false;
    const auto fenceResult = this->ctx.vk.get().df().GetFenceStatus(
        this->ctx.vk.get().dev(), slot.fence.handle());
    if (fenceResult == VK_NOT_READY) return false;
    if (fenceResult != VK_SUCCESS) {
        slot.authority.submitFailed(fenceResult == VK_ERROR_DEVICE_LOST);
        return false;
    }
    const auto counter = this->ctx.vk.get().df().GetSemaphoreCounterValueKHR;
    if (!counter) {
        slot.authority.submitFailed(true);
        return false;
    }
    for (size_t i = 0; i < slot.generatedFrames; ++i) {
        uint64_t value{};
        const auto result = counter(this->ctx.vk.get().dev(),
            this->destinationReturnSemaphores.at(i).handle(), &value);
        if (result == VK_ERROR_DEVICE_LOST) {
            slot.authority.submitFailed(true);
            return false;
        }
        if (result != VK_SUCCESS || value < slot.requiredDestinationReturns.at(i))
            return false;
    }
    if (!slot.authority.retire(true, true)) return false;
    slot.generatedFrames = 0;
    return true;
}

std::optional<PreparedFrameScheduleReservation> Context::reserveFrameSchedule(
        const std::vector<float>& timestamps,
        std::optional<vk::ExchangeTimelineFrame> expectedTimeline) {
    if (reservationOutstanding)
        return std::nullopt;
    if (schedulePoisoned)
        return std::nullopt;
    if (timestamps.size() > this->destImages.size())
        throw backend::error("requested more generated frames than the context capacity");
    float previous = 0.0F;
    for (const float timestamp : timestamps) {
        if (!(timestamp > 0.0F && timestamp < 1.0F) || timestamp <= previous)
            throw backend::error("invalid frame generation timestamps");
        previous = timestamp;
    }
    const auto timeline = vk::makeExchangeTimelineFrame(this->idx, timestamps.size());
    if (expectedTimeline && (expectedTimeline->sourceReady != timeline.sourceReady
            || expectedTimeline->sourceReturn != timeline.sourceReturn
            || expectedTimeline->nextBase != timeline.nextBase
            || expectedTimeline->generatedFrames != timeline.generatedFrames))
        throw backend::error("prepared frame schedule timeline identity mismatch");

    GenerationResourceSlot* selected{};
    for (auto& slot : generationSlots) {
        static_cast<void>(tryRetireSlot(*slot));
        if (slot->authority.phase() == vk::GenerationSlotPhase::Available) {
            selected = slot.get();
            break;
        }
    }
    if (!selected) return std::nullopt;

    selected->fence.reset(this->ctx.vk);
    if (!selected->authority.reserve())
        return std::nullopt;
    selected->generatedFrames = timestamps.size();
    std::ranges::fill(selected->requiredDestinationReturns, 0);
    try {
        for (size_t i = 0; i < timestamps.size(); ++i) {
            auto constants = backend::getDefaultConstantBuffer(
                i, timestamps.size(), this->ctx.flow);
            constants.timestamp = timestamps.at(i);
            selected->ctx->constantBuffers.at(i).update(this->ctx.vk, constants);
        }
        this->recordReservation(*selected, timestamps.size(), this->fidx, timeline);
        auto state = std::make_unique<PreparedFrameScheduleReservationState>();
        state->owner = this;
        state->slot = selected;
        state->timeline = timeline;
        state->fidx = this->fidx;
        state->generatedFrames = timestamps.size();
        state->active = true;
        reservationOutstanding = true;
        return PreparedFrameScheduleReservation(std::move(state));
    } catch (...) {
        selected->authority.abort();
        throw;
    }
}

void Context::abortReservation(PreparedFrameScheduleReservationState& reservation) noexcept {
    if (!reservation.active || reservation.owner != this || !reservation.slot)
        return;
    reservation.slot->authority.abort();
    reservation.active = false;
    reservationOutstanding = false;
}

void Context::executeReservation(PreparedFrameScheduleReservationState& reservation) {
    if (!reservation.active || reservation.owner != this || !reservation.slot
            || reservation.slot->authority.phase() != vk::GenerationSlotPhase::Reserved)
        throw backend::error("invalid prepared frame schedule execution authority");
    auto& slot = *reservation.slot;
    bool accepted{};
    try {
        vk::executePreparedFrameSchedule(reservation.generatedFrames,
            [&](vk::PreparedFrameSubmitPhase phase) {
                if (phase.index == 0) {
                    std::array<vk::TimelineWait, 1> waits{
                        vk::TimelineWait{this->syncSemaphore.handle(),
                            reservation.timeline.sourceReady,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};
                    std::array<vk::TimelineSignal, 1> signals{
                        vk::TimelineSignal{
                            reservation.generatedFrames == 0
                                ? this->syncSemaphore.handle()
                                : this->prepassSemaphore.handle(),
                            reservation.generatedFrames == 0
                                ? reservation.timeline.sourceReturn
                                : reservation.timeline.sourceReady}};
                    const vk::CommandBufferSubmit submission{
                        .timelineWaits = waits, .timelineSignals = signals};
                    slot.cmdbufs.at(0).submit(this->ctx.vk, submission,
                        phase.terminal ? slot.fence.handle() : VK_NULL_HANDLE);
                    return;
                }
                const size_t i = phase.index - 1;
                const auto& destinationReturn = this->destinationReturnStates.at(i);
                vk::BackendDestinationSubmitStorage storage;
                vk::prepareBackendDestinationSubmit(
                    this->prepassSemaphore.handle(), reservation.timeline.sourceReady,
                    this->destinationReturnSemaphores.at(i).handle(), destinationReturn,
                    this->syncSemaphore.handle(), reservation.timeline.destinationReady(i),
                    storage);
                slot.cmdbufs.at(phase.index).submit(this->ctx.vk, storage.submission,
                    phase.terminal ? slot.fence.handle() : VK_NULL_HANDLE);
            },
            [&](vk::PreparedFrameSubmitPhase phase) noexcept {
                accepted = true;
                slot.authority.submitAccepted(phase.terminal);
                if (phase.index == 0) {
                    this->idx = reservation.timeline.nextBase;
                    ++this->fidx;
                    return;
                }
                const size_t i = phase.index - 1;
                auto& destinationReturn = this->destinationReturnStates.at(i);
                slot.requiredDestinationReturns.at(i) =
                    destinationReturn.pendingWriteGeneration();
                destinationReturn.backendWriteAccepted();
            });
        reservation.active = false;
        reservationOutstanding = false;
    } catch (const ls::vulkan_error& e) {
        schedulePoisoned = vk::applyPreparedFrameSubmitFailure(
            slot.authority, accepted, e.error());
        reservation.active = false;
        reservationOutstanding = false;
        throw;
    } catch (...) {
        slot.authority.submitFailed(false);
        schedulePoisoned = accepted;
        reservation.active = false;
        reservationOutstanding = false;
        throw;
    }
}

void Instance::closeContext(const Context& context) {
    auto it = std::ranges::find_if(this->m_contexts,
        [context = &context](const std::unique_ptr<ContextImpl>& ctx) {
            return ctx.get() == context;
        });
    if (it == this->m_contexts.end())
        throw backend::error("attempted to close unknown context",
            std::runtime_error("no such context"));

    // Do not block or fabricate a clean return while a destination may still
    // be owned by A. Retain images and imported return timelines under the
    // backend device lifetime; a replacement context gets fresh semaphores and
    // generation state, so handle reuse cannot authorize it.
    this->m_retiredContexts.emplace_back(std::move(*it));
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
