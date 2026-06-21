#include "post_process/cpu/YoloDetCpuPostProcessor.hpp"

#include <stdexcept>

YoloDetCpuPostProcessor::YoloDetCpuPostProcessor(const PostProcessorConfig& config) {
    (void)config;
}

void YoloDetCpuPostProcessor::process(
    const TensorViewMap& engineOutputViews,
    std::vector<PostProcessOutput>& processedBatch,
    BaseLogger& logger,
    cudaStream_t stream
) {
    (void)engineOutputViews;
    (void)processedBatch;
    (void)logger;
    (void)stream;
    throw std::runtime_error("YOLO detection CPU postprocessing is not implemented yet");
}
