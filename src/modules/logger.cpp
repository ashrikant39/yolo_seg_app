#include "logger.hpp"
#include <iostream>
#include <stdexcept>

void Logger::assignStream(std::ios_base::openmode mode){
    _logStream = std::ofstream(_logPath, mode);

    if(_logStream.is_open()){
        std::cout<<"Logging to file: "<<_logPath<<"\n";
    }else{
        throw std::runtime_error("Failes to open file: " + _logPath.string());
    }
}

// Default constructor : Logging to a default file 
Logger::Logger(): 
    _loggerSeverity(nvinfer1::ILogger::Severity::kINFO),
    _logPath(LoggerOptions::DEFAULT_LOG_FILE){
        assignStream(std::ios_base::out);
    }


// Constructor : Logging to File 
Logger::Logger(const fs::path& fileName, nvinfer1::ILogger::Severity severity):
    _logPath(fileName),
    _loggerSeverity(severity){
        assignStream(std::ios_base::out);
}

// Logging function based on logger severity, default is set to kINFO
// severity is an enum, so treat severity levels as integers
void Logger::log(Severity severity, const char* message) noexcept
{
    if(severity <= _loggerSeverity)
    {
        if(severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR)
            _logStream << "[INTERNAL ERROR] " << message << std::endl;

        else if(severity == nvinfer1::ILogger::Severity::kERROR)
            _logStream << "[ERROR] " << message << std::endl;

        else if(severity == nvinfer1::ILogger::Severity::kWARNING)
            _logStream << "[WARNING] " << message << std::endl;
        
        else
            _logStream << "[INFO] " << message << std::endl;
    }
}


void Logger::logTensorDims(Severity Severity, const char* tensorName, const nvinfer1::Dims& tensorDims){
    
    _logStream << "Name: " << tensorName << "\tDims: " ;
    
    for(int i=0; i<tensorDims.nbDims; i++){

        if(i == 0){
            _logStream << '[';
        }

        _logStream << tensorDims.d[i];

        if(i<tensorDims.nbDims-1){
            _logStream << ", ";
        }
        else{
            _logStream << ']';
        }
    }

    _logStream << '\n';
}

Logger::~Logger() {
    log(nvinfer1::ILogger::Severity::kINFO, "Logger Exiting!");
}
