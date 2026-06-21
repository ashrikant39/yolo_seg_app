#pragma once

#include <functional>
#include <vector>

#include "core/tensor.hpp"
#include "enums.hpp"
#include "core/cuda.hpp"

struct TensorGroupConfig {
    TensorGroupList preProcessing;
    TensorGroupList inference;
    TensorGroupList postProcessing;
};

enum class TensorTransferKind {
    Copy,
    PrefetchToCpu,
    PrefetchToCuda
};

struct TensorTransfer {
    TensorGroup sourceGroup;
    TensorGroup targetGroup;
    std::reference_wrapper<TensorViewMap> sourceBufferViews;
    std::reference_wrapper<TensorViewMap> targetBufferViews;
    TensorTransferKind kind = TensorTransferKind::Copy;
};

struct PreProcessingTensorContext {
    std::reference_wrapper<TensorViewMap> bufferViews;
    TensorGroup group;
};

struct InferenceTensorContext {
    std::reference_wrapper<TensorViewMap> inputBufferViews;
    TensorGroup inputGroup;
    std::reference_wrapper<TensorViewMap> outputBufferViews;
    TensorGroup outputGroup;
    std::vector<std::reference_wrapper<TensorViewMap>> bindableTensorViews;
};

struct PostProcessingTensorContext {
    std::reference_wrapper<TensorViewMap> bufferViews;
    TensorGroup group;
};

struct PipelineTensorContext {
    PreProcessingTensorContext preProcessing;
    InferenceTensorContext inference;
    PostProcessingTensorContext postProcessing;
    std::vector<TensorTransfer> preProcessingToInference;
    std::vector<TensorTransfer> inferenceToPostProcessing;
};
