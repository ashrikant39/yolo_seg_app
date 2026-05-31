#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <mutex>

#include "core/utils.hpp"

namespace fs = std::filesystem;

class BaseLogger {

    public:
        explicit BaseLogger();
        explicit BaseLogger(const fs::path& fileName);
        explicit BaseLogger(const fs::path& fileName, LoggingSeverity severity);
        void log(LoggingSeverity severity, const char* msg) noexcept ;
        
        template <class ...Ts>
        void logConcatMessage(LoggingSeverity severity, Ts&&... xs){
            
            try {
                std::lock_guard<std::mutex> lock(m_loggerMutex);
                if(severity <= m_severity){

                    if(severity == LoggingSeverity::INTERNAL_ERROR)
                        m_logStream << "[INTERNAL ERROR] ";

                    else if(severity == LoggingSeverity::ERROR)
                        m_logStream << "[ERROR] ";

                    else if(severity == LoggingSeverity::WARNING)
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
        LoggingSeverity m_severity;
        std::filesystem::path m_logFilePath;
        std::ofstream m_logStream;
        std::mutex m_loggerMutex;
};