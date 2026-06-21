#include "post_process/gpu/YoloDetGpuPostProcessor.hpp"

#include <stdexcept>

YoloDetGpuPostProcessor::YoloDetGpuPostProcessor(const PostProcessorConfig& config) {
    (void)config;
}

void YoloDetGpuPostProcessor::process(
    const TensorViewMap& engineOutputViews,
    std::vector<PostProcessOutput>& processedBatch,
    BaseLogger& logger,
    cudaStream_t stream
) {
    (void)engineOutputViews;
    (void)processedBatch;
    (void)logger;
    (void)stream;
    throw std::runtime_error("YOLO detection GPU postprocessing is not implemented yet");
}
