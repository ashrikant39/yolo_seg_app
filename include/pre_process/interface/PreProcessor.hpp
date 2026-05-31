#pragma once

#include "source/utils/frame.hpp"
#include "core/tensor.hpp"

class PreProcessor {

    public:
        virtual ~PreProcessor() = default;
        virtual bool process(
            const BatchFrameData& inputData,
            TensorViewMap& outputMap,
            const std::vector<std::string>& inputKeys
        ) = 0;
};