#pragma once

#include <filesystem>
#include <string_view>
#include <vector>

#include "post_process/interface/PostProcessor.hpp"
#include "post_process/config/PostProcessorConfig.hpp"

namespace fs = std::filesystem;

struct YoloSegCpuPostProcessorSimpleSettings {
    static constexpr std::string_view BoxKey = "boxes";
    static constexpr std::string_view MaskKey = "masks";
    static constexpr std::string_view LabelKey = "classlabel";
    static constexpr std::string_view ScoreKey = "objectness";
};

/**
 * @brief CPU-side Simple post-processing for YOLO-seg style TensorRT outputs.
 *
 * Responsibilities:
 * - convert output tensor buffers from FP16 to FP32 host buffers,
 * - decode boxes/scores/mask coefficients,
 * - run NMS,
 * - generate and save segmentation outputs.
 */
class YoloSegCpuPostProcessorSimple : public PostProcessor {

    public:
        YoloSegCpuPostProcessorSimple(const PostProcessorConfig& config);

        void process(
            const TensorViewMap& engineOutputViews,
            std::vector<PostProcessOutput>& processedBatch,
            BaseLogger& logger,
            cudaStream_t stream
        ) override ;

    private:
        float m_confidenceThresh, m_iouThresh, m_maskThresh;
        size_t m_maxDetections;

};
