#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "core/yamlParser.hpp"
#include "application/Application.hpp"
#include "source/utils/frame.hpp"

std::vector<std::string> TensorKeys(const TensorViewMap& bufferViews) {
    std::vector<std::string> keys;
    keys.reserve(bufferViews.size());
    for (const auto& [name, tensor] : bufferViews) {
        keys.push_back(name);
    }
    return keys;
}

Application::Application(const std::filesystem::path& yamlPath):
    Application(loadAppSettingsFromYaml(yamlPath)) {}

Application::Application(const AppSettings& settings):
    m_settings(settings),
    m_baseLogger(settings.logFilePath, settings.loggingSeverity),
    m_memManager(TensorGroupConfig{
        .preProcessing = settings.preProcessingTensorGroups,
        .inference = settings.inferenceTensorGroups,
        .postProcessing = settings.postProcessingTensorGroups
    }) {

    FrameSourceConfig frameSourceCfg{
        .frameSourceType = settings.frameSourceType,
        .sourcePath = settings.frameSourcePath,
        .imgHeight = settings.origImgHeight,
        .imgWidth = settings.origImgWidth,
        .batchSize = settings.batchSize
    };

    PreProcessorConfig preprocessCfg{
        .modelType = settings.modelType,
        .preferredDevice = settings.preferredDevicePreProc,
        .imgRgbOrdering = settings.imgChannelOrdering,
        .imgResizeHeight = settings.imgPreProcessedImgH,
        .imgResizeWidth = settings.imgPreProcessedImgW,
        .numImgChannels = StaticSettings::NUM_IMG_CHANNELS,
        .outputDataType = settings.preprocessedDataType,
        .imgScalingFactor = settings.imgPreProcessScalingFactor,
        .imgMean = settings.imgPreProcessMeanFactor,
        .ndimsOfInputs = settings.ndimsOfInputs
    };

    InferenceBackendConfig inferCfg{
        .inferBackend = settings.inferenceBackendType,
        .modelType = settings.modelType,
        .processDevice = settings.preferredInferenceDevice,
        .modelFilePath = settings.serializedModelPath
    };

    PostProcessorConfig postprocessCfg{
        .outputSpecs = settings.outputTensorSpecs,
        .modelType = settings.modelType,
        .outputType = settings.outputType,
        .preferedDevice = settings.preferredDevicePostProc,
        .confThreshold = settings.confThreshold,
        .iouThreshold = settings.iouThreshold,
        .maskThreshold = settings.maskThreshold,
        .maxDetections = settings.maxDetections,
        .outputTensorStartLocs = settings.outputTensorStartLocs
    };

    ResultSinkConfig resultCfg{
        .sinkType = settings.resultSinkType,
        .saveMode = settings.saveDetMode,
        .drawDetectionMode = settings.drawDetMode,
        .lineThickness = settings.lineThickness
    };

    if (!settings.resultsDir.empty()) {
        std::error_code error;
        std::filesystem::create_directories(settings.resultsDir, error);
        if (error) {
            throw std::runtime_error(
                "Failed to create resultsDir: " + settings.resultsDir.string() +
                " (" + error.message() + ")"
            );
        }
    }

    m_frameSource = createFrameSource(frameSourceCfg);
    m_preProcessor = createPreProcessor(preprocessCfg);
    m_inferBackend = createInferenceBackend(inferCfg, m_baseLogger); // assert tensor names with tensor specs in yaml
    m_postProcessor = createPostProcessor(postprocessCfg);
    m_resultSink = createResultSink(resultCfg);

    m_memManager.allocateAllTensors(settings.inputTensorSpecs);
    m_memManager.allocateAllTensors(settings.outputTensorSpecs);
}

void Application::run() {

    auto bufferContext = m_memManager.createPipelineTensorContext();
    m_inferBackend->bindTensorViewMaps(bufferContext.inference.bindableTensorViews);

    const std::vector<std::string> inputKeys = TensorKeys(bufferContext.preProcessing.bufferViews.get());
    const std::vector<std::string> outputKeys = TensorKeys(bufferContext.postProcessing.bufferViews.get());

    using Clock = std::chrono::steady_clock;
    const Clock::time_point startTime = Clock::now();
    size_t totalSourceFrames = 0;
    size_t totalBatches = 0;
    size_t batchSize = m_settings.batchSize;

    std::vector<PostProcessOutput> processedBatch(batchSize);

    while (m_frameSource->readBatch(m_currBatch, m_baseLogger)) {

        for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            m_currBatch.metas[batchIdx].inputWidth = m_settings.imgPreProcessedImgW;
            m_currBatch.metas[batchIdx].inputHeight = m_settings.imgPreProcessedImgH;
            m_currBatch.metas[batchIdx].resultsDir = m_settings.resultsDir;
            processedBatch[batchIdx].metadata = m_currBatch.metas[batchIdx];
        }

        m_preProcessor->process(m_currBatch, bufferContext.preProcessing.bufferViews.get());

        totalSourceFrames += countSourceFrames(m_currBatch);
        ++totalBatches;

        CudaStream streamHolder;
        if (m_settings.preferredInferenceDevice == PreferredProcessingDevice::PREFER_GPU) {
            streamHolder.createStream();
        }
        cudaStream_t stream = streamHolder.get();

        MemoryManager::transferTensors(bufferContext.preProcessingToInference, inputKeys, stream);
        m_inferBackend->runInference(
            bufferContext.inference.inputBufferViews.get(),
            bufferContext.inference.outputBufferViews.get(),
            stream
        );
        MemoryManager::transferTensors(bufferContext.inferenceToPostProcessing, outputKeys, stream);

        if (stream) {
            CUDA_THROW(cudaStreamSynchronize(stream));
        }

        m_postProcessor->process(bufferContext.postProcessing.bufferViews.get(), processedBatch, m_baseLogger, stream);
        m_resultSink->consumeBatch(processedBatch, m_baseLogger);
    }

    const Clock::time_point endTime = Clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(endTime - startTime).count();
    const double totalFps = elapsedSeconds > 0.0
        ? static_cast<double>(totalSourceFrames) / elapsedSeconds
        : 0.0;

    m_baseLogger.logConcatMessage(
        LoggingSeverityType::INFO,
        "Total source frames processed: ", totalSourceFrames,
        ", total batches processed: ", totalBatches,
        ", elapsed seconds: ", elapsedSeconds,
        ", total FPS: ", totalFps,
        '\n'
    );
}
