#pragma once

#include "pre_process/config/PreProcessorConfig.hpp"
#include "pre_process/interface/PreProcessor.hpp"
#include "pre_process/modes/YoloSegCpuPreProcessor.hpp"


/**
 * @brief Create a preprocessor from configuration.
 * @param config Preprocessor configuration.
 * @return Owning pointer to the selected preprocessor.
 * @throws std::runtime_error for unsupported configurations.
 */
std::unique_ptr<PreProcessor> createPreProcessor(PreProcessorConfig config);
