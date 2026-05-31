#include "backends/modes/YoloSegTRTBackend.hpp"
#include "backends/utils/trtUtils.hpp"

YoloSegTRTBackend::YoloSegTRTBackend(
    const InferenceBackendConfig& config,
    BaseLogger& baseLogger
):
    m_logger(TrtLoggerAdaptor(baseLogger)),
    m_runtime(std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger))) {

        // Check if runtime is set
        if(!m_runtime){
            m_logger.log(Severity::kERROR, "Failed to create TensorRT runtime.");
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }
        // Deserialize the model (in .engine file) and create an engine for running inference
        // Load the engine file onto a std::vector buffer

        std::vector<char> engineData = readEngineFileToArray(config.modelFilePath);

        // Deserialize the engineData to an Engine using the Runtime created
        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            m_runtime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()
            )
        );
        
        if(!m_engine){   
            m_logger.log(Severity::kERROR, "Failed to create TensorRT engine.");
            throw std::runtime_error("Failed to create TensorRT engine.");
        }

        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

        if(!m_context){   
            m_logger.log(Severity::kERROR, "Failed to create Execution Context for engine.");
            throw std::runtime_error("Failed to create Execution Context for engine.");
        }
}

TensorInfoMap YoloSegTRTBackend::getTensorInfos() {

    TensorInfoMap infoMap;
    
    for (const auto& mode : SupportedTensorModes) {
        for (const std::string& name : getTensorNames(mode) ) {
            infoMap.emplace(
                name,
                TensorInfo{
                    fromTrtDims(m_engine->getTensorShape(name)),
                    fromTrtDType(m_engine->getTensorDataType(name)),
                    mode
                }
            )
        }
    }

    return infoMap;
}


bool YoloSegTRTBackend::runInference(
    const TensorViewMap& inputTensors,
    TensorViewMap& outputTensors,
    cudaStream_t stream
) {
    
    if (!(
        tensorMapInDevice(inputTensors, DeviceType::CUDA) ||
        tensorMapInDevice(inputTensors, DeviceType::Unified)
    )) {
        throw std::runtime_error("All Input tensors must either be allocated in either GPU Mem or Unified Mem.");
    }

    if (!(
        tensorMapInDevice(outputTensors, DeviceType::CUDA) ||
        tensorMapInDevice(outputTensors, DeviceType::Unified)
    )) {
        throw std::runtime_error("All Output tensors must either be allocated in either GPU Mem or Unified Mem.");
    }

    NVTX_RANGE("EnqueueV3");
    if ( !m_context->enqueueV3(stream) ) {
        m_logger.log(Severity::kERROR, "EnqueueV3 Failed.");
        throw std::runtime_error("EnqueueV3 Failed: \n");
    }
    NVTX_POP();

    return true;
}