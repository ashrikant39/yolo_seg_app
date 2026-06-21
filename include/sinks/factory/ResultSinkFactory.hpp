#pragma once

#include "sinks/config/ResultSinkConfig.hpp"
#include "sinks/interface/ResultSink.hpp"

std::unique_ptr<ResultSink> createResultSink(const ResultSinkConfig& config);