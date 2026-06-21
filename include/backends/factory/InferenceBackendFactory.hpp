#pragma once

#include "backends/config/InferenceBackendConfig.hpp"
#include "backends/interface/InferenceBackend.hpp"
#include "logging/BaseLogger.hpp"

std::unique_ptr<InferenceBackend> createInferenceBackend(InferenceBackendConfig config, BaseLogger& baseLogger);