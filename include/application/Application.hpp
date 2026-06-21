#pragma once

#include <filesystem>

#include "AppSettings.hpp"
#include "source/config/FrameSourceConfig.hpp"
#include "backends/config/InferenceBackendConfig.hpp"
#include "logging/BaseLogger.hpp"
#include "memory_management/MemoryManager.hpp"
#include "pre_process/config/PreProcessorConfig.hpp"
#include "post_process/config/PostProcessorConfig.hpp"
#include "sinks/config/ResultSinkConfig.hpp"

#include "source/factory/FrameSourceFactory.hpp"
#include "backends/factory/InferenceBackendFactory.hpp"
#include "pre_process/factory/PreProcessorFactory.hpp"
#include "post_process/factory/PostProcessFactory.hpp"
#include "sinks/factory/ResultSinkFactory.hpp"


class Application {

    public:
        explicit Application(const std::filesystem::path& yamlPath);
        void run();

    private:
        explicit Application(const AppSettings& settings);

        AppSettings m_settings;
        BaseLogger m_baseLogger;
        MemoryManager m_memManager;
        std::unique_ptr<FrameSource> m_frameSource;
        std::unique_ptr<PreProcessor> m_preProcessor;
        std::unique_ptr<InferenceBackend> m_inferBackend;
        std::unique_ptr<PostProcessor> m_postProcessor;
        std::unique_ptr<ResultSink> m_resultSink;
        BatchFrameData m_currBatch;
};
