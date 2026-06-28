#pragma once

#include <NvInfer.h>

#include "logging/BaseLogger.hpp"

using TrtSeverity = nvinfer1::ILogger::Severity;

/**
 * @brief Converts TensorRT logger severity to the application logger severity.
 */
inline LoggingSeverityType getBaseLoggerSeverity(TrtSeverity severity) noexcept {

    switch (severity) {

        case TrtSeverity::kINFO:
            return LoggingSeverityType::INFO;
        
        case TrtSeverity::kERROR:
            return LoggingSeverityType::ERROR;
        
        case TrtSeverity::kINTERNAL_ERROR:
            return LoggingSeverityType::INTERNAL_ERROR;
        
        case TrtSeverity::kWARNING:
            return LoggingSeverityType::WARNING;
        
        case TrtSeverity::kVERBOSE:
            return LoggingSeverityType::VERBOSE;
        
        default:
            return LoggingSeverityType::INTERNAL_ERROR;
    }
}

/**
 * @brief Adapter that routes TensorRT logger callbacks into BaseLogger.
 */
class TrtLoggerAdaptor : public nvinfer1::ILogger {

    public:
        /**
         * @brief Creates an adapter over an existing application logger.
         */
        explicit TrtLoggerAdaptor(BaseLogger& logger):
            m_baseLogger(logger) {
        }

        /**
         * @brief TensorRT logger callback implementation.
         */
        void log(TrtSeverity severity, const char *msg) noexcept override {
            m_baseLogger.log(
                getBaseLoggerSeverity(severity),
                msg
            );
        }

        /**
         * @brief Writes one concatenated TensorRT-scoped log message.
         */
        template <class ...Ts>
        void logConcatMessage(TrtSeverity severity, Ts&&... xs) {
            m_baseLogger.logConcatMessage(
                getBaseLoggerSeverity(severity),
                std::forward<Ts>(xs)...
            );
        }

        /**
         * @brief Logs a TensorRT tensor shape in a compact human-readable format.
         */
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
            m_baseLogger.log(getBaseLoggerSeverity(severity), ss.str().c_str());
            
        }

    private:
        BaseLogger& m_baseLogger;
        
};
