#include <iostream>

#include "logging/BaseLogger.hpp"

namespace {
constexpr const char* DEFAULT_LOG_FILE = "main.log";
constexpr LoggingSeverityType DEFAULT_LOG_SEVERITY = LoggingSeverityType::INFO;
} // namespace

bool BaseLogger::assignStream(){

    m_logStream = std::ofstream(m_logFilePath, std::ios_base::out);

    if(m_logStream.is_open()){
        std::cout << "Logging to file: " << m_logFilePath << '\n';

    } else {
        throw std::runtime_error("Failed to open file: " + m_logFilePath.string());
    }

    return true;
}

BaseLogger::BaseLogger():
    m_severity(DEFAULT_LOG_SEVERITY),
    m_logFilePath(DEFAULT_LOG_FILE) {
        assignStream();
    }

BaseLogger::BaseLogger(const fs::path& fileName):
    m_severity(DEFAULT_LOG_SEVERITY),
    m_logFilePath(fileName) {
        assignStream();
    }

BaseLogger::BaseLogger(const fs::path& fileName, LoggingSeverityType severity):
    m_severity(severity),
    m_logFilePath(fileName) {
        assignStream();
    }

void BaseLogger::log(LoggingSeverityType severity, const char* msg) noexcept {
    
    try {

        std::lock_guard<std::mutex> lock(m_loggerMutex);
        
        if(severity >= m_severity) {
            if(severity == LoggingSeverityType::INTERNAL_ERROR)
                m_logStream << "[INTERNAL ERROR] " << msg << '\n';

            else if(severity == LoggingSeverityType::ERROR)
                m_logStream << "[ERROR] " << msg << '\n';

            else if(severity == LoggingSeverityType::WARNING)
                m_logStream << "[WARNING] " << msg << '\n';
            
            else
                m_logStream << "[INFO] " << msg << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Unknown logger failure \n" << e.what();
    }
        
}
