#pragma once

#include "backends/config/InferenceBackendConfig.hpp"
#include "backends/interface/InferenceBackend.hpp"
#include "logging/BaseLogger.hpp"

/**
 * @brief Create an inference backend from configuration.
 * @param config Backend configuration.
 * @param baseLogger Logger shared with backend implementations.
 * @return Owning pointer to the selected backend.
 * @throws std::runtime_error for unsupported backend configurations.
 */
std::unique_ptr<InferenceBackend> createInferenceBackend(InferenceBackendConfig config, BaseLogger& baseLogger);
