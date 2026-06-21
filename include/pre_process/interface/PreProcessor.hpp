#pragma once

#include "source/utils/frame.hpp"
#include "core/tensor.hpp"

class PreProcessor {

    public:
        virtual ~PreProcessor() = default;
        virtual void process(
            const BatchFrameData& inputData,
            TensorViewMap& resultBufferViews
        ) = 0;
};
