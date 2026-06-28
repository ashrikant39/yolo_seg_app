#pragma once

/**
 * @brief Severity levels used by the application logger.
 */
enum class LoggingSeverityType {
    VERBOSE,
    INFO,
    WARNING,
    ERROR,
    INTERNAL_ERROR,

    kVERBOSE = VERBOSE,
    kINFO = INFO,
    kWARNING = WARNING,
    kERROR = ERROR,
    kINTERNAL_ERROR = INTERNAL_ERROR
};

/**
 * @brief Short alias for logger severity.
 */
using Severity = LoggingSeverityType;
