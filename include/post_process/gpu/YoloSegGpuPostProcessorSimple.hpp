#pragma once

#include <vector>
#include <filesystem>

#include "post_process/interface/PostProcessor.hpp"
#include "post_process/config/PostProcessorConfig.hpp"

namespace fs = std::filesystem;

/**
 * @brief CPU-side Simple post-processing for YOLO-seg style TensorRT outputs.
 *
 * Responsibilities:
 * - convert output tensors from FP16 to FP32 host buffers,
 * - decode boxes/scores/mask coefficients,
 * - run NMS,
 * - generate and save segmentation outputs.
 */
class YoloSegGpuPostProcessorSimple : public PostProcessor {

    public:
        YoloSegGpuPostProcessorSimple(const PostProcessorConfig& config) {

        }

        void process(
            const TensorViewMap& engineOutputBatch,
            std::vector<PostProcessOutput>& processedBatch,
            Logger& logger,
            cudaStream_t stream
        ) override {
            
        }
};
