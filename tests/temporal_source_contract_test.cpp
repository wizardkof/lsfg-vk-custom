#include "lsfg-vk-backend/lsfgvk.hpp"

#include <cassert>
#include <type_traits>

using namespace lsfgvk::backend;

int main() {
    static_assert(temporalSourceSlotIndex(TemporalSourceSlot::Slot0) == 0);
    static_assert(temporalSourceSlotIndex(TemporalSourceSlot::Slot1) == 1);
    static_assert(!std::is_copy_constructible_v<RuntimeGenerateDiagnosticPending>);
    static_assert(std::is_move_constructible_v<RuntimeGenerateDiagnosticPending>);
    static_assert(!std::is_copy_constructible_v<RuntimeGenerationOperationRetirement>);
    static_assert(std::is_move_constructible_v<RuntimeGenerationOperationRetirement>);
#ifdef LSFGVK_TESTING_SHADOW_SPLIT
    using ShadowSubmit = RuntimeShadowIngestSnapshot (Instance::*)(
        RuntimeGenerateSession&, VkImage, vk::SyncFdPayload,
        TemporalSourceSlot, uint64_t);
    using ShadowRetire = RuntimeShadowIngestSnapshot (Instance::*)(
        RuntimeGenerateSession&);
    using ShadowGenerate = RuntimeShadowGenerateSnapshot (Instance::*)(
        RuntimeGenerateSession&, RuntimeTemporalPairIdentity);
    using ShadowGenerateRetire = RuntimeShadowGenerateSnapshot (Instance::*)(
        RuntimeGenerateSession&);
    const ShadowSubmit shadowSubmit = &Instance::submitShadowTemporalIngest;
    const ShadowRetire shadowRetire = &Instance::retireShadowTemporalIngest;
    const ShadowGenerate shadowGenerate = &Instance::submitShadowPrepassGenerate;
    const ShadowGenerateRetire shadowGenerateRetire =
        &Instance::retireShadowPrepassGenerate;
    assert(shadowSubmit != nullptr && shadowRetire != nullptr
        && shadowGenerate != nullptr && shadowGenerateRetire != nullptr);
#endif

    const RuntimeTemporalPairIdentity bc{
        TemporalSourceSlot::Slot1, TemporalSourceSlot::Slot0, 2, 3, 1};
    const RuntimeTemporalPairIdentity cd{
        TemporalSourceSlot::Slot0, TemporalSourceSlot::Slot1, 3, 4, 2};
    const RuntimeTemporalPairIdentity de{
        TemporalSourceSlot::Slot1, TemporalSourceSlot::Slot0, 4, 5, 3};
    assert(validTemporalPair(bc) && temporalPairFrameIndex(bc) == 0);
    assert(validTemporalPair(cd) && temporalPairFrameIndex(cd) == 1);
    assert(validTemporalPair(de) && temporalPairFrameIndex(de) == 0);

    auto invalid = bc;
    invalid.newerSlot = invalid.olderSlot;
    assert(!validTemporalPair(invalid));
    invalid = bc;
    invalid.newerFrameId = 0;
    assert(!validTemporalPair(invalid));
}
