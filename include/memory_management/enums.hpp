#pragma once

#include <vector>

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
using TensorGroupList = std::vector<TensorGroup>;
