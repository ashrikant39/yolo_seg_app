#include "post_process/cpu/YoloSegCpuPostProcessorRaw.hpp"

#include <stdexcept>

YoloSegCpuPostProcessorRaw::YoloSegCpuPostProcessorRaw(const PostProcessorConfig& config) {
    (void)config;
}

void YoloSegCpuPostProcessorRaw::process(
    const TensorViewMap& engineOutputViews,
    std::vector<PostProcessOutput>& processedBatch,
    BaseLogger& logger,
    cudaStream_t stream
) {
    (void)engineOutputViews;
    (void)processedBatch;
    (void)logger;
    (void)stream;
    throw std::runtime_error("Raw YOLO segmentation CPU postprocessing is not implemented yet");
}
