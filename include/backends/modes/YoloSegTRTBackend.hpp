#pragma once

#include <NvInfer.h>
#include <array>

#include "logging/BackendLoggers/TrtLoggerAdaptor.hpp"
#include "backends/interface/InferenceBackend.hpp"
#include "backends/config/InferenceBackendConfig.hpp"
#include "backends/utils/trtUtils.hpp"


inline constexpr std::array<nvinfer1::TensorIOMode, 2> SupportedTensorModes{
    nvinfer1::TensorIOMode::kINPUT,
    nvinfer1::TensorIOMode::kOUTPUT
};


/**
 * @brief TensorRT inference backend for YOLO segmentation engines.
 */
class YoloSegTRTBackend : public InferenceBackend {

    public:
        /**
         * @brief Deserialize a TensorRT engine and create an execution context.
         * @param config Backend configuration containing the engine path.
         * @param baseLogger Logger used through the TensorRT logger adaptor.
         */
        YoloSegTRTBackend(
            const InferenceBackendConfig& config,
            BaseLogger& baseLogger
        );

        /**
         * @copydoc InferenceBackend::bindTensorViewMap
         */
        void bindTensorViewMap(const TensorViewMap& bufferViews) override;

        /**
         * @copydoc InferenceBackend::runInference
         */
        bool runInference(
            const TensorViewMap& inputBufferViews,
            TensorViewMap& outputBufferViews,
            cudaStream_t stream
        ) override;

        /**
         * @copydoc InferenceBackend::getTensorSpecs
         */
        TensorSpecMap getTensorSpecs() override;

    private:
        TrtLoggerAdaptor m_logger;
        std::unique_ptr<nvinfer1::IRuntime> m_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context;
};
