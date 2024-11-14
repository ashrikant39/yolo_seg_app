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

        // Default constructor : Logging to stdout 
        // Since std::cout is not owned by the program, but std::ofstream can be,
        // logStream is a simple pointer to the stream, whereas fileStream is a unique pointer
        // to the file stream.

        Logger();

        // Constructor : Logging to File 
        Logger(const std::string&, NVLogger::Severity severity = NVLogger::Severity::kINFO);

        // Constructor : Logging to a stringstream
        // Since the stream has to be modified, const ref is not used here.
        Logger(std::stringstream& stream, NVLogger::Severity severity = NVLogger::Severity::kINFO);

        // Commonly logging convntions have the order: severity, msg
        // Severity is an enum class with kINTERNAL_ERROR = 0 , 
        // kERROR = 1 , kWARNING = 2 , kINFO = 3
        void log(NVLogger::Severity, const char*) noexcept override;

        // Get Logger Severity
        NVLogger::Severity getLoggerSeverity()
        {
            return this->loggerSeverity;
        }

        bool assertStreamOwnership()
        {
            return this->ownTheStream;
        }

        // Destructor
        ~Logger();
    
    private:
        NVLogger::Severity loggerSeverity;
        bool ownTheStream;
        std::unique_ptr<std::ofstream> fileStream;
        std::ostream* logStream;

};