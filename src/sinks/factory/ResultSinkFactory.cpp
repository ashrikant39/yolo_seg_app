#include "sinks/factory/ResultSinkFactory.hpp"
#include "sinks/modes/DrawDetectionSink.hpp"
#include "sinks/modes/FileDetectionSink.hpp"

std::unique_ptr<ResultSink> createResultSink(const ResultSinkConfig& config) {

    switch (config.sinkType) {

        case ResultSinkType::SAVE_DETECTIONS:
            return std::make_unique<FileDetectionSink>(config.saveMode == SaveDetectionMode::NORMALIZED);

        case ResultSinkType::DRAW_DETCTIONS:
            return std::make_unique<DrawDetectionSink>(config);

        default:
            throw std::runtime_error("Unsupported Result Sink mode.");
    }
}