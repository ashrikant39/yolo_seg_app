#include "pre_process/factory/PreProcessorFactory.hpp"

std::unique_ptr<PreProcessor> createPreProcessor(PreProcessorConfig config) {
    
    if (
        config.modelType == ModelType::YOLO_SEGMENTATION &&
        config.preferredDevice == ProcessDevice::PREFER_CPU
    ) {
        return std::make_unique<YoloSegCpuPreProcessor>(config);
    }

    throw std::runtime_error("Unsupported configuration for preprocessor");
}