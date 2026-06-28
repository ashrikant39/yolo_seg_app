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


/**
 * @brief Top-level YOLO segmentation application.
 *
 * Application owns the configured frame source, preprocessing stage, inference
 * backend, postprocessor, result sink, logger, and memory manager. It is
 * normally constructed from a YAML file and then executed with run().
 */
class Application {

    public:
        /**
         * @brief Build the full application from a YAML configuration file.
         * @param yamlPath Path to a YAML file accepted by loadAppSettingsFromYaml().
         * @throws std::runtime_error if configuration, output directories, model,
         *         or component construction fails.
         */
        explicit Application(const std::filesystem::path& yamlPath);

        /**
         * @brief Execute the full source -> preprocess -> inference -> postprocess -> sink loop.
         *
         * The loop processes full batches from the configured FrameSource and logs
         * aggregate throughput when the source is exhausted.
         */
        void run();

    private:
        /**
         * @brief Build the application from already parsed settings.
         * @param settings Complete application settings.
         */
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
