#include "backends/factory/InferenceBackendFactory.hpp"
#include "backends/interface/InferenceBackend.hpp"
#include "backends/modes/YoloSegTRTBackend.hpp"


std::unique_ptr<InferenceBackend> createInferenceBackend(InferenceBackendConfig config, BaseLogger& baseLogger) {

    if (config.inferBackend == BackendType::YoloSegTRT) {
        return std::make_unique<YoloSegTRTBackend>(config, baseLogger);
    }

    return nullptr;
}