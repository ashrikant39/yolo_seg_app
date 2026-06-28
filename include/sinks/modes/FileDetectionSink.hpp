#pragma once

#include "sinks/interface/ResultSink.hpp"
#include "sinks/config/ResultSinkConfig.hpp"


/**
 * @brief Result sink that serializes detections to binary files.
 */
class FileDetectionSink : public ResultSink {

    public:
        /**
         * @brief Construct a file sink.
         * @param saveNormalized Whether detections should be normalized before saving.
         */
        FileDetectionSink(bool saveNormalized);
        
        /**
         * @copydoc ResultSink::consumeSingle
         */
        void consumeSingle(PostProcessOutput& output, BaseLogger& logger) override;

    private:
        bool m_saveNormalized;
};
