#pragma once

#include <vector>

/**
 * @brief Named tensor memory groups used by preprocessing, inference, and postprocessing.
 */
enum class TensorGroup {
    HostInput,
    PinnedInput,
    DeviceInput,
    UnifiedInput,

    HostOutput,
    PinnedOutput,
    DeviceOutput,
    UnifiedOutput,

    HostPostProcessOutput,
    DevicePostProcessOutput
};
/**
 * @brief Ordered list of tensor groups loaded from YAML.
 */
using TensorGroupList = std::vector<TensorGroup>;
