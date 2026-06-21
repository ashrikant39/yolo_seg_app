#pragma once

#include <functional>
#include <vector>

#include "core/tensor.hpp"

class InferenceBackend {

    public:
        virtual ~InferenceBackend() = default;

        virtual bool runInference(
            const TensorViewMap& inputBufferViews,
            TensorViewMap& outputBufferViews,
            cudaStream_t stream
        ) = 0;

        virtual void bindTensorViewMap(const TensorViewMap& bufferViews) = 0;
        void bindTensorViewMaps(const std::vector<std::reference_wrapper<TensorViewMap>>& bufferViewMaps) {
            for (const std::reference_wrapper<TensorViewMap>& bufferViewMap : bufferViewMaps) {
                bindTensorViewMap(bufferViewMap.get());
            }
        }

        virtual TensorSpecMap getTensorSpecs() = 0;
};
