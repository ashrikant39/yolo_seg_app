#include "logger.h"

// Default constructor : Logging to stdout 
Logger::Logger(): 
    loggerSeverity(NVLogger::Severity::kINFO),
    ownTheStream(false), 
    fileStream(nullptr),
    logStream(&std::cout) 
    {}



// Constructor : Logging to File 
Logger::Logger(const std::string& fileName, NVLogger::Severity severity) : 
    loggerSeverity(severity),
    ownTheStream(true), 
    fileStream(nullptr)
{
    this->fileStream = std::make_unique<std::ofstream>(fileName, std::ios::out |  std::ios::app);
    
    if(fileStream->is_open())
    {
        this->logStream = fileStream.get();
        std::cout<<"Logging to file: "<<fileName<<"\n";
    }
    else
    {
        std::cout<<"Unable to open the file: "<<fileName<<"\n";
        std::cout<<"Logging directly to console!"<<std::endl;
        this->ownTheStream = false;
        this->logStream = &std::cout;
    }
}




// Constructor : Logging to a stringstream
Logger::Logger(std::stringstream& stream, NVLogger::Severity severity) : 
    loggerSeverity(severity),
    ownTheStream(false), 
    fileStream(nullptr),
    logStream(&stream)
    {}


// Logging function based on logger severity, default is set to kINFO
// severity is an enum, so treat severity levels as integers
void Logger::log(NVLogger::Severity severity, const char* message) noexcept
{
    if(severity <= this->loggerSeverity)
    {
        if(severity == NVLogger::Severity::kINTERNAL_ERROR)
            *this->logStream << "[INTERNAL ERROR] " <<message << std::endl;

        else if(severity == NVLogger::Severity::kERROR)
            *this->logStream << "[ERROR] " <<message << std::endl;

        else if(severity == NVLogger::Severity::kWARNING)
            *this->logStream << "[WARNING] " <<message << std::endl;
        
        else
            *this->logStream << "[INFO] " <<message << std::endl;
    }
}

Logger::~Logger() {
    if (ownTheStream && fileStream && fileStream->is_open()) 
    {
        fileStream->close();
    }
}
