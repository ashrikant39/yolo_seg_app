#pragma once

#include "pre_process/config/PreProcessorConfig.hpp"
#include "pre_process/interface/PreProcessor.hpp"
#include "pre_process/modes/YoloSegCpuPreProcessor.hpp"


std::unique_ptr<PreProcessor> createPreProcessor(PreProcessorConfig config);