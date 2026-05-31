#pragma once

#include <filesystem>

#include "sinks/interface/ResultSink.hpp"
#include "sinks/config/ResultSinkConfig.hpp"

namespace fs = std::filesystem;

class DrawDetectionSink : public ResultSink {

    public:
        DrawDetectionSink(const ResultSinkConfig& config);
        void consumeSingle(PostProcessOutput& output, Logger& logger) override;

    private:
        bool m_drawBoxes, m_drawMasks, m_drawContours;
        int m_lineThickness;
        // bool m_drawContourBasedMasks;
};