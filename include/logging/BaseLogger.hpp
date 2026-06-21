#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <mutex>

#include "logging/enums.hpp"

namespace fs = std::filesystem;

class BaseLogger {

    public:
        explicit BaseLogger();
        explicit BaseLogger(const fs::path& fileName);
        explicit BaseLogger(const fs::path& fileName, LoggingSeverityType severity);
        void log(LoggingSeverityType severity, const char* msg) noexcept ;
        
        template <class ...Ts>
        void logConcatMessage(LoggingSeverityType severity, Ts&&... xs) {
            
            try {
                std::lock_guard<std::mutex> lock(m_loggerMutex);
                if(severity >= m_severity){

                    if(severity == LoggingSeverityType::INTERNAL_ERROR)
                        m_logStream << "[INTERNAL ERROR] ";

                    else if(severity == LoggingSeverityType::ERROR)
                        m_logStream << "[ERROR] ";

                    else if(severity == LoggingSeverityType::WARNING)
                        m_logStream << "[WARNING] ";
                    
                    else
                        m_logStream << "[INFO] ";

                    (m_logStream << ... << std::forward<Ts>(xs));
                }

            } catch (const std::exception& e) {
                std::cerr << "Unknown logger failure \n" << e.what();
            }
        }

    private:
        bool assignStream();

    private:
        LoggingSeverityType m_severity;
        std::filesystem::path m_logFilePath;
        std::ofstream m_logStream;
        std::mutex m_loggerMutex;
};
