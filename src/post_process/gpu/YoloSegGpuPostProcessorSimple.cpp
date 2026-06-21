#include "post_process/gpu/YoloSegGpuPostProcessorSimple.hpp"

#include <stdexcept>

YoloSegGpuPostProcessorSimple::YoloSegGpuPostProcessorSimple(const PostProcessorConfig& config) {
    (void)config;
}

void YoloSegGpuPostProcessorSimple::process(
    const TensorViewMap& engineOutputViews,
    std::vector<PostProcessOutput>& processedBatch,
    BaseLogger& logger,
    cudaStream_t stream
) {
    (void)engineOutputViews;
    (void)processedBatch;
    (void)logger;
    (void)stream;
    throw std::runtime_error("Modified YOLO segmentation GPU postprocessing is not implemented yet");
}
