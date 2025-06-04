#pragma once

#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>

using namespace nvinfer1;

class Logger : public ILogger
{
    public: 

        // Default constructor : Logging to "main.log" 
        // fileStream is a unique pointer to the file stream.
        Logger();

        // Constructor : Logging to File 
        Logger(const std::string&, ILogger::Severity severity = ILogger::Severity::kINFO);

        // Commonly logging conventions have the order: severity, msg
        // Severity is an enum class with kINTERNAL_ERROR = 0 , 
        // kERROR = 1 , kWARNING = 2 , kINFO = 3
        void log(ILogger::Severity, const char*) noexcept override;

        // Get Logger Severity
        ILogger::Severity getLoggerSeverity(){
            return m_loggerSeverity;
        }

        // Destructor
        ~Logger();
    
    private:
        ILogger::Severity m_loggerSeverity;
        std::ofstream m_logStream;
};