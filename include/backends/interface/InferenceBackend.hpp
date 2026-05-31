#pragma once

#include "core/tensor.hpp"

class InferenceBackend {

    public:
        virtual ~Inference() = default;

        virtual bool runInference(
            const TensorViewMap& inputTensors,
            TensorViewMap& outputTensors,
            cudaStream_t stream
        ) = 0;

        virtual TensorInfoMap getTensorInfos() = 0;
};