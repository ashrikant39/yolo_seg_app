#pragma once

#include <filesystem>

#include "sinks/interface/ResultSink.hpp"
#include "sinks/config/ResultSinkConfig.hpp"

namespace fs = std::filesystem;

/**
 * @brief Result sink that draws detections on source images.
 */
class DrawDetectionSink : public ResultSink {

    public:
        /**
         * @brief Construct from result sink drawing configuration.
         * @param config Drawing mode and line thickness.
         */
        DrawDetectionSink(const ResultSinkConfig& config);

        /**
         * @copydoc ResultSink::consumeSingle
         */
        void consumeSingle(PostProcessOutput& output, BaseLogger& logger) override;

    private:
        bool m_drawBoxes, m_drawMasks, m_drawContours;
        int m_lineThickness;
};
