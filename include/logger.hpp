#pragma once

#include <NvInfer.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include "utils/options.hpp"
#include <utility>

using Severity = nvinfer1::ILogger::Severity;
namespace fs = std::filesystem;

/**
 * @brief File-backed logger implementing TensorRT's ILogger interface.
 *
 * The logger writes severity-tagged messages to a configured log file.
 */
class Logger : public nvinfer1::ILogger
{
    public: 

        /**
         * @brief Construct a logger using the default log path.
         */
        explicit Logger();

        /**
         * @brief Construct a logger with explicit path and severity threshold.
         */
        explicit Logger(const fs::path& fileName, Severity severity = Severity::kINFO);

        /**
         * @brief TensorRT logger callback.
         * @param severity Message severity.
         * @param message Null-terminated message.
         */
        void log(Severity severity, const char* message) noexcept override;

        /**
         * @brief Stream-concatenate arbitrary values into the log with severity prefix.
         */
        template <class ...Ts>
        void logConcatMessage(Severity severity, Ts&&... xs){
            if(severity <= m_loggerSeverity){
                if(severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR)
                    m_logStream << "[INTERNAL ERROR] ";

                else if(severity == nvinfer1::ILogger::Severity::kERROR)
                    m_logStream << "[ERROR] ";

                else if(severity == nvinfer1::ILogger::Severity::kWARNING)
                    m_logStream << "[WARNING] ";
                
                else
                    m_logStream << "[INFO] ";
            }

            (m_logStream << ... << std::forward<Ts>(xs));
        }

        /**
         * @brief Log a tensor name and its dimensions.
         */
        void logTensorDims(Severity Severity, const char* tensorName, const nvinfer1::Dims& tensorDims);

        /**
         * @brief Get configured severity threshold.
         */
        [[nodiscard]] Severity getLoggerSeverity(){
            return m_loggerSeverity;
        }

        /**
         * @brief Get the currently configured log file path.
         */
        [[nodiscard]] std::filesystem::path getLogFilePath(){
            return m_logPath;
        }

        /**
         * @brief Destructor writes shutdown message and closes stream.
         */
        ~Logger();
    
    private:
        Severity m_loggerSeverity;
        std::filesystem::path m_logPath;
        std::ofstream m_logStream;

        void assignStream(std::ios_base::openmode mode);
};
