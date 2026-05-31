#pragma once


#include "post_process/config/PostProcessorConfig.hpp"
#include "post_process/interface/PostProcessor.hpp"


inline ModelType getModelTypeFromNames(const std::vector<std::string>& outputNames) {

    if (outputNames.empty()) {
        throw std::runtime_error("Config has output names missing");
    }
    if ( outputNames.size() == 1 ) {
        return ModelType::YOLO_DETECTION;

    } else {
        return ModelType::YOLO_SEGMENTATION;
    }

}

inline OutputType getOutputTypeFromNames(const std::vector<std::string>& outputNames) {

    if (outputNames.empty()) {
        throw std::runtime_error("Config has output names missing");
    }

    if ( outputNames.size() == 1 ) {
        return OutputType::YOLO_RAW_DETECTION;
    }

    for (const std::string& name : outputNames) {

        std::string::size_type idx = name.find("output");

        if ( idx != std::string::npos && std::isdigit(name.at(idx + 6)) ) {
            return OutputType::YOLO_RAW_SEGMENTATION;
        }
    }

    return OutputType::YOLO_MODIFIED_SEGMENTATION;
}


void validateParameters(PostProcessorConfig& config);


std::unique_ptr<PostProcessor> createPostProcessor(PostProcessorConfig config);