#pragma once

#include <functional>
#include <vector>

#include "core/tensor.hpp"
#include "enums.hpp"
#include "core/cuda.hpp"

struct GroupLists {
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
    TensorGroupType sourceGroup;
    TensorGroupType targetGroup;
    std::reference_wrapper<TensorViewMap> sourceTensors;
    std::reference_wrapper<TensorViewMap> targetTensors;
    TensorTransferKind kind = TensorTransferKind::Copy;
};

struct PreProcessingContext {
    std::reference_wrapper<TensorViewMap> tensors;
    TensorGroupType group;
};

struct InferenceContext {
    std::reference_wrapper<TensorViewMap> inputTensors;
    TensorGroupType inputGroup;
    std::reference_wrapper<TensorViewMap> outputTensors;
    TensorGroupType outputGroup;
    std::vector<std::reference_wrapper<TensorViewMap>> bindableTensorMaps;
};

struct PostProcessingContext {
    std::reference_wrapper<TensorViewMap> tensors;
    TensorGroupType group;
};

struct PipelineTensorContext {
    PreProcessingContext preProcessing;
    InferenceContext inference;
    PostProcessingContext postProcessing;
    std::vector<TensorTransfer> preProcessingToInference;
    std::vector<TensorTransfer> inferenceToPostProcessing;
};
