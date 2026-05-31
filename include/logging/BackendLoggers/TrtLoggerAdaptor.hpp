#pragma once

#include <NvInfer.h>

#include "logging/BaseLogger.hpp"

using TrtSeverity = nvinfer1::ILogger::Severity;

inline LoggingSeverity getBaseLoggerSeverity(TrtSeverity severity) noexcept {

    switch (severity) {

        case TrtSeverity::kINFO:
            return LoggingSeverity::INFO;
        
        case TrtSeverity::kERROR:
            return LoggingSeverity::ERROR;
        
        case TrtSeverity::kINTERNAL_ERROR:
            return LoggingSeverity::INTERNAL_ERROR;
        
        case TrtSeverity::kWARNING:
            return LoggingSeverity::WARNING;
        
        case TrtSeverity::kVERBOSE:
            return LoggingSeverity::VERBOSE;
        
        default:
            return LoggingSeverity::INTERNAL_ERROR;
    }
}

class TrtLoggerAdaptor : public nvinfer1::ILogger {

    public:
        explicit TrtLoggerAdaptor(BaseLogger& logger):
            m_baseLogger(logger) {
        }

        void log(TrtSeverity severity, const char *msg) noexcept override {
            m_baseLogger.log(
                getBaseLoggerSeverity(severity),
                msg
            );
        }

        template <class ...Ts>
        void logConcatMessage(TrtSeverity severity, Ts&&... xs) {
            m_baseLogger.logConcatMessage(
                getBaseLoggerSeverity(severity),
                std::forward<Ts>(xs)...
            );
        }

        void logTensorDims(TrtSeverity severity,
            const std::string& tensorName,
            const nvinfer1::Dims& tensorDims
        ) {

            std::stringstream ss;
            ss << "Name: " << tensorName.c_str() << "\t Shape: ";

            for(int i=0; i<tensorDims.nbDims; i++){
                if(i == 0){
                    ss << '[';
                }
                ss << tensorDims.d[i];
                if(i<tensorDims.nbDims-1){
                    ss << ", ";
                }
                else{
                    ss << ']';
                }
            }
            m_baseLogger.log(severity, ss.str().c_str());
            
        }

    private:
        BaseLogger& m_baseLogger;
        
};