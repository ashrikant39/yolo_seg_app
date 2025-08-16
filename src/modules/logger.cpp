#include "logger.h"
#include <iostream>
#include <stdexcept>
#include <sstream>

void Logger::assignStream(std::ios_base::openmode mode){
    m_logStream = std::ofstream(m_logPath, mode);

    if(m_logStream.is_open()){
        std::cout<<"Logging to file: "<<m_logPath<<"\n";
    }else{
        throw std::runtime_error("Failes to open file: " + m_logPath.string());
    }
}

// Default constructor : Logging to stdout 
Logger::Logger(): 
    m_loggerSeverity(nvinfer1::ILogger::Severity::kINFO),
    m_logPath("main.log"){
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


void Logger::logTensorDims(Severity severity, const char* tensorName, const nvinfer1::Dims& tensorDims){
    
    std::stringstream ss;
    ss << "Tensor Name: " << tensorName << "\t Tensor Dims: [";
    
    int i;
    
    for(i=0; i<tensorDims.nbDims - 1; i++){
        ss << tensorDims.d[i] << ", ";
    }

    ss << tensorDims.d[i] << "]";
    log(severity, ss.str().c_str());
    
}

Logger::~Logger() {
    log(nvinfer1::ILogger::Severity::kINFO, "Logger Exiting!");
}
