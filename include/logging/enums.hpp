#pragma once

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

using Severity = LoggingSeverityType;
