#pragma once

#include "sinks/interface/ResultSink.hpp"
#include "sinks/config/ResultSinkConfig.hpp"


class FileDetectionSink : public ResultSink {

    public:
        FileDetectionSink(bool saveNormalized);
        
        void consumeSingle(PostProcessOutput& output, BaseLogger& logger) override;

    private:
        bool m_saveNormalized;
};