#pragma once

#include "sinks/config/ResultSinkConfig.hpp"
#include "sinks/interface/ResultSink.hpp"

/**
 * @brief Creates the configured result sink implementation.
 *
 * @param config Sink settings loaded from YAML.
 * @return Owning pointer to the selected sink.
 */
std::unique_ptr<ResultSink> createResultSink(const ResultSinkConfig& config);
