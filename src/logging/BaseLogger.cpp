#include <iostream>

#include "logging/BaseLogger.hpp"
#include "AppSettings.hpp"

bool BaseLogger::assignStream(){

    m_logStream = std::ofstream(m_logFilePath, std::ios_base::out);

    if(m_logStream.is_open()){
        std::cout << "Logging to file: " << m_logFilePath << '\n';

    } else {
        throw std::runtime_error("Failed to open file: " + m_logFilePath.string());
    }
}

BaseLogger::BaseLogger():
    m_severity(DefaultSettings::DEFAULT_LOG_SEVERITY),
    m_logFilePath(DefaultSettings::DEFAULT_LOG_FILE) {
        assignStream();
    }

BaseLogger::BaseLogger(const fs::path& fileName):
    m_severity(DefaultSettings::DEFAULT_LOG_SEVERITY),
    m_logFilePath(fileName) {
        assignStream();
    }

BaseLogger::BaseLogger(const fs::path& fileName, LoggingSeverity severity):
    m_severity(severity),
    m_logFilePath(fileName) {
        assignStream();
    }

void BaseLogger::log(LoggingSeverity severity, const char* msg) noexcept {
    
    try {

        std::lock_guard<std::mutex> lock(m_loggerMutex);
        
        if(severity <= m_severity) {
            if(severity == LoggingSeverity::INTERNAL_ERROR)
                m_logStream << "[INTERNAL ERROR] " << msg << '\n';

            else if(severity == LoggingSeverity::ERROR)
                m_logStream << "[ERROR] " << msg << '\n';

            else if(severity == LoggingSeverity::WARNING)
                m_logStream << "[WARNING] " << msg << '\n';
            
            else
                m_logStream << "[INFO] " << msg << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Unknown logger failure \n" << e.what();
    }
        
}