#pragma once

#include <functional>
#include <vector>

#include "core/tensor.hpp"
#include "enums.hpp"
#include "core/cuda.hpp"

/**
 * @brief Tensor group lists for each pipeline stage.
 */
struct TensorGroupConfig {
    TensorGroupList preProcessing;
    TensorGroupList inference;
    TensorGroupList postProcessing;
};

/**
 * @brief Transfer/readiness operation between tensor groups.
 */
enum class TensorTransferKind {
    Copy,
    PrefetchToCpu,
    PrefetchToCuda
};

/**
 * @brief Description of a tensor transfer or unified-memory prefetch.
 */
struct TensorTransfer {
    TensorGroup sourceGroup;
    TensorGroup targetGroup;
    std::reference_wrapper<TensorViewMap> sourceBufferViews;
    std::reference_wrapper<TensorViewMap> targetBufferViews;
    TensorTransferKind kind = TensorTransferKind::Copy;
};

/**
 * @brief Tensor views selected for preprocessing.
 */
struct PreProcessingTensorContext {
    std::reference_wrapper<TensorViewMap> bufferViews;
    TensorGroup group;
};

/**
 * @brief Tensor views selected for inference input/output and backend binding.
 */
struct InferenceTensorContext {
    std::reference_wrapper<TensorViewMap> inputBufferViews;
    TensorGroup inputGroup;
    std::reference_wrapper<TensorViewMap> outputBufferViews;
    TensorGroup outputGroup;
    std::vector<std::reference_wrapper<TensorViewMap>> bindableTensorViews;
};

/**
 * @brief Tensor views selected for postprocessing.
 */
struct PostProcessingTensorContext {
    std::reference_wrapper<TensorViewMap> bufferViews;
    TensorGroup group;
};

/**
 * @brief All tensor contexts and transfer operations needed by Application::run().
 */
struct PipelineTensorContext {
    PreProcessingTensorContext preProcessing;
    InferenceTensorContext inference;
    PostProcessingTensorContext postProcessing;
    std::vector<TensorTransfer> preProcessingToInference;
    std::vector<TensorTransfer> inferenceToPostProcessing;
};
