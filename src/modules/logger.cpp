#include "logger.h"

// Default constructor : Logging to stdout 
Logger::Logger(): 
    m_loggerSeverity(NVLogger::Severity::kINFO),
    m_logStream(std::ofstream("main.log", std::ios::out))
    {}


// Constructor : Logging to File 
Logger::Logger(const std::string& fileName, NVLogger::Severity severity): 
    m_loggerSeverity(severity),
    m_logStream(std::ofstream(fileName, std::ios::out |  std::ios::app)){    

    if(m_logStream.is_open()){

        std::cout<<"Logging to file: "<<fileName<<"\n";
    
    }else{

        std::cout<<"Unable to open the file: "<<fileName<<"\n";
        std::cout<<"Logging directly to console!"<<std::endl;
    }
}

// Logging function based on logger severity, default is set to kINFO
// severity is an enum, so treat severity levels as integers
void Logger::log(NVLogger::Severity severity, const char* message) noexcept
{
    if(severity <= m_loggerSeverity)
    {
        if(severity == NVLogger::Severity::kINTERNAL_ERROR)
            m_logStream << "[INTERNAL ERROR] " << message << std::endl;

        else if(severity == NVLogger::Severity::kERROR)
            m_logStream << "[ERROR] " << message << std::endl;

        else if(severity == NVLogger::Severity::kWARNING)
            m_logStream << "[WARNING] " << message << std::endl;
        
        else
            m_logStream << "[INFO] " << message << std::endl;
    }
}

Logger::~Logger() {
    if (m_logStream.is_open()){
        m_logStream.close();
    }
}
