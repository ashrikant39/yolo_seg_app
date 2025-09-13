#pragma once

#include <NvInfer.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include "options.hpp"
#include <utility>

using Severity = nvinfer1::ILogger::Severity;
namespace fs = std::filesystem;

class Logger : public nvinfer1::ILogger
{
    public: 

        // Default constructor : Logging to "main.log" 
        explicit Logger();

        // Constructor : Logging to File 
        explicit Logger(const fs::path& fileName, Severity severity = Severity::kINFO);

        // Commonly logging conventions have the order: severity, msg
        // Severity is an enum class with kINTERNAL_ERROR = 0 , 
        // kERROR = 1 , kWARNING = 2 , kINFO = 3
        void log(Severity severity, const char* message) noexcept override;

        template <class ...Ts>
        void logConcatMessage(Severity severity, Ts&&... xs){
            if(severity <= _loggerSeverity){
                if(severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR)
                    _logStream << "[INTERNAL ERROR] ";

                else if(severity == nvinfer1::ILogger::Severity::kERROR)
                    _logStream << "[ERROR] ";

                else if(severity == nvinfer1::ILogger::Severity::kWARNING)
                    _logStream << "[WARNING] ";
                
                else
                    _logStream << "[INFO] ";
            }

            (_logStream << ... << std::forward<Ts>(xs));
        }

        void logTensorDims(Severity Severity, const char* tensorName, const nvinfer1::Dims& tensorDims);

        // Get Logger Severity
        [[nodiscard]] Severity getLoggerSeverity(){
            return _loggerSeverity;
        }

        [[nodiscard]] std::filesystem::path getLogFilePath(){
            return _logPath;
        }

        // Destructor
        ~Logger();
    
    private:
        Severity _loggerSeverity;
        std::filesystem::path _logPath;
        std::ofstream _logStream;

        void assignStream(std::ios_base::openmode mode);
};
