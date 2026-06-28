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

/**
 * @brief Thread-safe file logger used by application, backend, and pipeline code.
 */
class BaseLogger {

    public:
        /**
         * @brief Creates a logger with default output settings.
         */
        explicit BaseLogger();
        /**
         * @brief Creates a logger that writes to the provided file path.
         */
        explicit BaseLogger(const fs::path& fileName);
        /**
         * @brief Creates a logger with an explicit output path and severity threshold.
         */
        explicit BaseLogger(const fs::path& fileName, LoggingSeverityType severity);
        /**
         * @brief Writes one log message when the severity passes the configured threshold.
         */
        void log(LoggingSeverityType severity, const char* msg) noexcept ;
        
        /**
         * @brief Concatenates and writes arbitrary streamable values as one log message.
         *
         * @tparam Ts Streamable argument types.
         * @param severity Message severity.
         * @param xs Values to append to the log stream.
         */
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
        /**
         * @brief Opens the configured log stream.
         */
        bool assignStream();

    private:
        LoggingSeverityType m_severity;
        std::filesystem::path m_logFilePath;
        std::ofstream m_logStream;
        std::mutex m_loggerMutex;
};
