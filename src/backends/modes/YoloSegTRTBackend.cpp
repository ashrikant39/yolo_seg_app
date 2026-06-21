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
            m_logger.log(TrtSeverity::kERROR, "Failed to create TensorRT runtime.");
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
            m_logger.log(TrtSeverity::kERROR, "Failed to create TensorRT engine.");
            throw std::runtime_error("Failed to create TensorRT engine.");
        }

        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

        if(!m_context){
            m_logger.log(TrtSeverity::kERROR, "Failed to create Execution Context for engine.");
            throw std::runtime_error("Failed to create Execution Context for engine.");
        }
}

TensorSpecMap YoloSegTRTBackend::getTensorSpecs() {

    TensorSpecMap infoMap;

    for (const auto& mode : SupportedTensorModes) {
        for (const std::string& name : getTensorNames(m_engine, mode) ) {
            infoMap.emplace(
                name,
                TensorSpec {
                    TrtDims2Shape(m_engine->getTensorShape(name.c_str())),
                    TrtType2DataType(m_engine->getTensorDataType(name.c_str())),
                    TrtIOMode2IOMode(mode)
                }
            );
        }
    }

    return infoMap;
}

void YoloSegTRTBackend::bindTensorViewMap(const TensorViewMap& bufferViews) {

    for (const auto& [name, tv] : bufferViews) {
        if (tv.memoryType != MemoryType::CudaMem && tv.memoryType != MemoryType::Unified) {
            continue;
        }

        if (!tv.data) {
            throw std::runtime_error("Memory not allocated for tensor: " + name);
        }
        // cudaPointerAttributes attributes;
        // cudaError_t error = cudaPointerGetAttributes(&attributes, tv.data);

        // switch (attributes.type) {
        //     case cudaMemoryTypeUnregistered:
        //         std::cout << "Host memory (unregistered)" << std::endl;
        //         break;
        //     case cudaMemoryTypeHost:
        //         std::cout << "Host memory (pinned/allocated via cudaHostAlloc)" << std::endl;
        //         break;
        //     case cudaMemoryTypeDevice:
        //         std::cout << "CUDA Device memory (allocated via cudaMalloc)" << std::endl;
        //         break;
        //     case cudaMemoryTypeManaged:
        //         std::cout << "Unified / Managed memory (accessible by both CPU and GPU)" << std::endl;
        //         break;
        // }

        m_context->setTensorAddress(name.c_str(), tv.data);
    }
}


bool YoloSegTRTBackend::runInference(
    const TensorViewMap& inputBufferViews,
    TensorViewMap& outputBufferViews,
    cudaStream_t stream
) {

    if (!(
        TensorViewsOnDevice(inputBufferViews, DeviceType::CUDA) ||
        TensorViewsInMemory(inputBufferViews, MemoryType::Unified)
    )) {
        throw std::runtime_error("All input buffer views must reference either GPU memory or unified memory.");
    }

    if (!(
        TensorViewsOnDevice(outputBufferViews, DeviceType::CUDA) ||
        TensorViewsInMemory(outputBufferViews, MemoryType::Unified)
    )) {
        throw std::runtime_error("All output buffer views must reference either GPU memory or unified memory.");
    }

    if ( !m_context->enqueueV3(stream) ) {
        m_logger.log(TrtSeverity::kERROR, "EnqueueV3 Failed.");
        throw std::runtime_error("EnqueueV3 Failed: \n");
    }

    return true;
}
