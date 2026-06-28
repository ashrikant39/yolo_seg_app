#pragma once

#include <functional>
#include <vector>

#include "core/tensor.hpp"

/**
 * @brief Abstract interface for model inference backends.
 *
 * Implementations bind tensor views by model tensor name and execute inference
 * on already prepared input/output memory.
 */
class InferenceBackend {

    public:
        virtual ~InferenceBackend() = default;

        /**
         * @brief Run inference using previously bound tensors.
         * @param inputBufferViews Input tensor views selected by MemoryManager.
         * @param outputBufferViews Output tensor views selected by MemoryManager.
         * @param stream CUDA stream for GPU backends, or nullptr for CPU-compatible paths.
         * @return true when inference is launched successfully.
         */
        virtual bool runInference(
            const TensorViewMap& inputBufferViews,
            TensorViewMap& outputBufferViews,
            cudaStream_t stream
        ) = 0;

        /**
         * @brief Bind a single tensor view map to the backend model context.
         * @param bufferViews Tensor views keyed by model tensor name.
         */
        virtual void bindTensorViewMap(const TensorViewMap& bufferViews) = 0;

        /**
         * @brief Bind multiple tensor view maps to the backend model context.
         * @param bufferViewMaps Reference wrappers to tensor maps selected for binding.
         */
        void bindTensorViewMaps(const std::vector<std::reference_wrapper<TensorViewMap>>& bufferViewMaps) {
            for (const std::reference_wrapper<TensorViewMap>& bufferViewMap : bufferViewMaps) {
                bindTensorViewMap(bufferViewMap.get());
            }
        }

        /**
         * @brief Query tensor specifications from the backend model.
         * @return Tensor specs keyed by backend tensor name.
         */
        virtual TensorSpecMap getTensorSpecs() = 0;
};
