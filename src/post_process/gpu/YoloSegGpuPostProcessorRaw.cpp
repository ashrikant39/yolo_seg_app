#include "post_process/gpu/YoloSegGpuPostProcessorRaw.hpp"

#include <stdexcept>

YoloSegGpuPostProcessorRaw::YoloSegGpuPostProcessorRaw(const PostProcessorConfig& config) {
    (void)config;
}

void YoloSegGpuPostProcessorRaw::process(
    const TensorViewMap& engineOutputViews,
    std::vector<PostProcessOutput>& processedBatch,
    BaseLogger& logger,
    cudaStream_t stream
) {
    (void)engineOutputViews;
    (void)processedBatch;
    (void)logger;
    (void)stream;
    throw std::runtime_error("Raw YOLO segmentation GPU postprocessing is not implemented yet");
}
