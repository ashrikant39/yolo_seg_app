#pragma once

#include <vector>

enum class TensorGroup {
    HostInput,          // CPU input
    PinnedInput,        // Pinned input
    DeviceInput,        // Cuda Input
    UnifiedInput,       // Unified Mem Input

    HostOutput,          // CPU Output
    PinnedOutput,        // Pinned Output
    DeviceOutput,        // Cuda Output
    UnifiedOutput,       // Unified Mem Output

    HostPostProcessOutput, // CPU Post processing Tensors
    DevicePostProcessOutput // Cuda Post processing Tensors
};

using SupportedTensorGroups = std::vector<TensorGroup>;

