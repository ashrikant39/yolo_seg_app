#include <cctype> 

#include "post_process/factory/PostProcessFactory.hpp"

#include "post_process/cpu/YoloDetCpuPostProcessor.hpp"
#include "post_process/cpu/YoloSegCpuPostProcessorRaw.hpp"
#include "post_process/cpu/YoloSegCpuPostProcessorSimple.hpp"


#include "post_process/gpu/YoloDetGpuPostProcessor.hpp"
#include "post_process/gpu/YoloSegGpuPostProcessorRaw.hpp"
#include "post_process/gpu/YoloSegGpuPostProcessorSimple.hpp"

void validateParameters(PostProcessorConfig& config) {

    const std::vector<std::string> outputNames = getOutputTensorNames(config.outputSpecs);

    if (config.modelType == ModelType::UNSET) {
        config.modelType = getModelTypeFromNames(outputNames);
    }

    if (config.outputType == OutputType::UNSET) {
        config.outputType = getOutputTypeFromNames(outputNames);
    }

    if (
        config.modelType == ModelType::YOLO_DETECTION &&
        config.outputType != OutputType::YOLO_RAW_DETECTION

    ) {
        throw std::runtime_error("Yolo-Detection model does not support modified outputs.");
    }

    if (config.outputType == OutputType::YOLO_MODIFIED_SEGMENTATION) {
        for (const auto& [name, start] : config.outputTensorStartLocs) {
            if (start != 0) {
                throw std::runtime_error(
                    "For modified outputs, tensor '" + name +
                    "' must start at its base address. Offset: " + std::to_string(start)
                );
            }
        }
    }
}

std::unique_ptr<PostProcessor> createPostProcessor(PostProcessorConfig config) {

    validateParameters(config);

    if (
        config.modelType == ModelType::YOLO_DETECTION &&
        config.preferedDevice == PreferredProcessingDevice::PREFER_CPU
    ) {

        return std::make_unique<YoloDetCpuPostProcessor>(config);
    }

    if ( 
        config.modelType == ModelType::YOLO_SEGMENTATION && 
        config.outputType == OutputType::YOLO_MODIFIED_SEGMENTATION &&
        config.preferedDevice == PreferredProcessingDevice::PREFER_CPU
    ) {

        return std::make_unique<YoloSegCpuPostProcessorSimple>(config);
    }

    if (
        config.modelType == ModelType::YOLO_SEGMENTATION &&
        config.outputType == OutputType::YOLO_RAW_SEGMENTATION &&
        config.preferedDevice == PreferredProcessingDevice::PREFER_CPU
    ) {

        return std::make_unique<YoloSegCpuPostProcessorRaw>(config);
    }

    if (
        config.modelType == ModelType::YOLO_DETECTION &&
        config.preferedDevice == PreferredProcessingDevice::PREFER_GPU
    ) {

        return std::make_unique<YoloDetGpuPostProcessor>(config);
    }

    if ( 
        config.modelType == ModelType::YOLO_SEGMENTATION && 
        config.outputType == OutputType::YOLO_MODIFIED_SEGMENTATION &&
        config.preferedDevice == PreferredProcessingDevice::PREFER_GPU
    ) {

        return std::make_unique<YoloSegGpuPostProcessorSimple>(config);
    }

    if ( 
        config.modelType == ModelType::YOLO_SEGMENTATION && 
        config.outputType == OutputType::YOLO_RAW_SEGMENTATION &&
        config.preferedDevice == PreferredProcessingDevice::PREFER_GPU
    ) {

        return std::make_unique<YoloSegGpuPostProcessorRaw>(config);
    }

    throw std::runtime_error("Unsupported postprocessor configuration");
}
