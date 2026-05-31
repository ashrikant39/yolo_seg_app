#include <cctype> 

#include "post_process/factory/PostProcessFactory.hpp"

#include "post_process/cpu/YoloDetCpuPostProcessor.hpp"
#include "post_process/cpu/YoloSegCpuPostProcessorRaw.hpp"
#include "post_process/cpu/YoloSegCpuPostProcessorSimple.hpp"


#include "post_process/gpu/YoloDetGpuPostProcessor.hpp"
#include "post_process/gpu/YoloSegGpuPostProcessorRaw.hpp"
#include "post_process/gpu/YoloSegGpuPostProcessorSimple.hpp"

void validateParameters(PostProcessorConfig& config) {

    if (config.modelType == ModelType::UNSET) {
        config.modelType = getModelTypeFromNames(config.outputNames);
    }

    if (config.outputType == OutputType::UNSET) {
        config.outputType = getOutputTypeFromNames(config.outputNames);
    }

    if (
        config.modelType == ModelType::YOLO_DETECTION &&
        config.outputType == OutputType::YOLO_RAW_DETECTION

    ) {
        throw std::runtime_error("Yolo-Detection model does not support modified outputs.");
    }

    if (config.outputType == OutputType::YOLO_MODIFIED_SEGMENTATION) {

        if(config.boxStart != 0) {
            throw std::runtime_error(
                "For modified outputs, all required tensors must have start of the tensor at its base address. \
                Provided boxes at an offset " +
                std::to_string(config.boxStart) 
            );
        };
        if(config.maskStart != 0) {
            throw std::runtime_error(
                "For modified outputs, all required tensors must have start of the tensor at its base address. \
                Provided masks at an offset " +
                std::to_string(config.maskStart) 
            );
        };
        if(config.classStart != 0) {
            throw std::runtime_error(
                "For modified outputs, all required tensors must have start of the tensor at its base address. \
                Provided class labels at an offset " +
                std::to_string(config.classStart) 
            );
        };
        if(config.objectnessStart != 0) {
            throw std::runtime_error(
                "For modified outputs, all required tensors must have start of the tensor at its base address. \
                Provided objectness at an offset " +
                std::to_string(config.objectnessStart) 
            );
        };
    }


}

std::unique_ptr<PostProcessor> createPostProcessor(PostProcessorConfig config) {

    validateParameters(config);

    if (
        config.modelType == ModelType::YOLO_DETECTION &&
        config.preferedDevice == ProcessDevice::PREFER_CPU
    ) {

        return std::make_unique<YoloDetCpuPostProcessor>(config);
    }

    if ( 
        config.modelType == ModelType::YOLO_SEGMENTATION && 
        config.outputType == OutputType::YOLO_MODIFIED_SEGMENTATION &&
        config.preferedDevice == ProcessDevice::PREFER_CPU
    ) {

        return std::make_unique<YoloSegCpuPostProcessorSimple>(config);
    }

    if (
        config.modelType == ModelType::YOLO_SEGMENTATION &&
        config.outputType == OutputType::YOLO_RAW_SEGMENTATION &&
        config.preferedDevice == ProcessDevice::PREFER_CPU
    ) {

        return std::make_unique<YoloSegCpuPostProcessorRaw>(config);
    }

    if (
        config.modelType == ModelType::YOLO_DETECTION &&
        config.preferedDevice == ProcessDevice::PREFER_GPU
    ) {

        return std::make_unique<YoloDetGpuPostProcessor>(config);
    }

    if ( 
        config.modelType == ModelType::YOLO_SEGMENTATION && 
        config.outputType == OutputType::YOLO_MODIFIED_SEGMENTATION &&
        config.preferedDevice == ProcessDevice::PREFER_GPU
    ) {

        return std::make_unique<YoloSegGpuPostProcessorSimple>(config);
    }

    if ( 
        config.modelType == ModelType::YOLO_SEGMENTATION && 
        config.outputType == OutputType::YOLO_RAW_SEGMENTATION &&
        config.preferedDevice == ProcessDevice::PREFER_GPU
    ) {

        return std::make_unique<YoloSegGpuPostProcessorRaw>(config);
    }

    throw std::runtime_error("Unsupported postprocessor configuration");
}