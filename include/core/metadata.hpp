#pragma once

#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

struct LoggerOptions {
    static constexpr const char* DEFAULT_LOG_FILE = "main.log";
};

struct FrameMetadata {

    uint64_t frameId = 0;
    uint64_t timestampNs = 0;

    // Important paths
    fs::path imagePath;
    fs::path resultsDir;
    fs::path saveDetPath;
    std::string saveMaskDirName;

    // Original image geometry
    size_t originalWidth = 0;
    size_t originalHeight = 0;
    size_t originalChannels = 0;

    // Network input geometry
    size_t inputWidth = 0;
    size_t inputHeight = 0;

    // Network output geometry
    size_t outputWidth = 0;
    size_t outputHeight = 0;
    
};

// For measurements
inline uint64_t nowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()
                .time_since_epoch()).count();
}

// For Logging
inline uint64_t wallClockNowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::system_clock::now()
                .time_since_epoch()).count();
}