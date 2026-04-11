#pragma once

#include "utils/tensor.hpp"
#include "logger.hpp"
#include <vector>
#include <filesystem>
#include "utils/options.hpp"
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

/**
 * @brief CPU-side post-processing for YOLO-seg style TensorRT outputs.
 *
 * Responsibilities:
 * - convert output tensors from FP16 to FP32 host buffers,
 * - decode boxes/scores/mask coefficients,
 * - run NMS,
 * - generate and save segmentation outputs.
 */
class PostProcessor{

    public:

        /**
         * @brief Construct post-processor with output directory and model input shape.
         * @param resultsDir Directory for saved masks/visualizations.
         * @param modelInputWidth Model input width in pixels.
         * @param modelInputHeight Model input height in pixels.
         */
        PostProcessor(
            const fs::path& resultsDir,
            int imageWidth,
            int imageHeight
        );

        void postProcessOutputs(
            CudaTensorMap& modelOutputMap,
            const std::vector<fs::path>& batchFileNames,
            Logger& logger);

    private:
        TensorMap m_postProcessTensorMap;
        fs::path m_resultsDir;
        int m_imageW{0};
        int m_imageH{0};
};