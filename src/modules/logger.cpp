#include "logger.hpp"
#include <iostream>
#include <stdexcept>

void Logger::assignStream(std::ios_base::openmode mode){
    m_logStream = std::ofstream(m_logPath, mode);

    if(m_logStream.is_open()){
        std::cout<<"Logging to file: "<<m_logPath<<"\n";
    }else{
        throw std::runtime_error("Failes to open file: " + m_logPath.string());
    }
}

// Default constructor : Logging to a default file 
Logger::Logger(): 
    m_loggerSeverity(nvinfer1::ILogger::Severity::kINFO),
    m_logPath(LoggerOptions::DEFAULT_LOG_FILE){
        assignStream(std::ios_base::out);
    }


// Constructor : Logging to File 
Logger::Logger(const fs::path& fileName, nvinfer1::ILogger::Severity severity):
    m_logPath(fileName),
    m_loggerSeverity(severity){
        assignStream(std::ios_base::out);
}

// Logging function based on logger severity, default is set to kINFO
// severity is an enum, so treat severity levels as integers
void Logger::log(Severity severity, const char* message) noexcept
{
    if(severity <= m_loggerSeverity)
    {
        if(severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR)
            m_logStream << "[INTERNAL ERROR] " << message << std::endl;

        else if(severity == nvinfer1::ILogger::Severity::kERROR)
            m_logStream << "[ERROR] " << message << std::endl;

        else if(severity == nvinfer1::ILogger::Severity::kWARNING)
            m_logStream << "[WARNING] " << message << std::endl;
        
        else
            m_logStream << "[INFO] " << message << std::endl;
    }
}


void Logger::logTensorDims(Severity Severity, const char* tensorName, const nvinfer1::Dims& tensorDims){
    
    m_logStream << "Name: " << tensorName << "\tDims: " ;
    
    for(int i=0; i<tensorDims.nbDims; i++){

        if(i == 0){
            m_logStream << '[';
        }

        m_logStream << tensorDims.d[i];

        if(i<tensorDims.nbDims-1){
            m_logStream << ", ";
        }
        else{
            m_logStream << ']';
        }
    }

    m_logStream << '\n';
}

Logger::~Logger() {
    log(nvinfer1::ILogger::Severity::kINFO, "Logger Exiting!");
}
