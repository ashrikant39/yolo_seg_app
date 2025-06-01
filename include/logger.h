#pragma once

#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>

using NVLogger = nvinfer1::ILogger;

class Logger : public NVLogger
{
    public: 

        // Default constructor : Logging to "main.log" 
        // fileStream is a unique pointer to the file stream.
        Logger();

        // Constructor : Logging to File 
        Logger(const std::string&, NVLogger::Severity severity = NVLogger::Severity::kINFO);

        // Commonly logging conventions have the order: severity, msg
        // Severity is an enum class with kINTERNAL_ERROR = 0 , 
        // kERROR = 1 , kWARNING = 2 , kINFO = 3
        void log(NVLogger::Severity, const char*) noexcept override;

        // Get Logger Severity
        NVLogger::Severity getLoggerSeverity(){
            return m_loggerSeverity;
        }

        // Destructor
        ~Logger();
    
    private:
        NVLogger::Severity m_loggerSeverity;
        std::ofstream m_logStream;
};