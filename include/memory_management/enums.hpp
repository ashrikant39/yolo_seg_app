#pragma once

#include <vector>

enum class TensorGroupType {
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
using TensorGroupList = std::vector<TensorGroupType>;
