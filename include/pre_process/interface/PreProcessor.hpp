#pragma once

#include "source/utils/frame.hpp"
#include "core/tensor.hpp"

/**
 * @brief Abstract interface for converting source frames into model input tensors.
 */
class PreProcessor {

    public:
        virtual ~PreProcessor() = default;

        /**
         * @brief Preprocess a batch of frames into output tensor views.
         * @param inputData Batch images and metadata from FrameSource.
         * @param resultBufferViews Output tensor views keyed by tensor name.
         */
        virtual void process(
            const BatchFrameData& inputData,
            TensorViewMap& resultBufferViews
        ) = 0;
};
