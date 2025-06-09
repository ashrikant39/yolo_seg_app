#include "pipeline.h"

std::vector<char> readEngineFileToArray(const std::string& fileName){

    std::ifstream file(fileName, std::ios::binary | std::ios::ate);

    if(!file.is_open()){
        throw std::runtime_error("Failed to open engine file: " + fileName);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);

    if(!file.read(engineData.data(), size)){
        throw std::runtime_error("Failed to read engine file: " + fileName);
    }

    return engineData;

}

size_t getElementSize(DataType dtype){

    size_t bytesPerElement;
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: bytesPerElement = 4; break;
        case nvinfer1::DataType::kHALF: bytesPerElement = 2; break;
        case nvinfer1::DataType::kBF16: bytesPerElement = 2; break;
        case nvinfer1::DataType::kINT8: bytesPerElement = 1; break;
        case nvinfer1::DataType::kINT32: bytesPerElement = 4; break;
        default: throw std::runtime_error("Unknown data type");
    }
    return bytesPerElement;

}


void InferencePipeline::allocateIODeviceMemory(){

    for(int32_t i=0; i < m_engine->getNbIOTensors(); i++){
        
        const char* name = m_engine->getIOTensorName(i);
        const Dims& tensorDims = m_engine->getTensorShape(name);

        size_t totalTensorSize = getElementSize(m_engine->getTensorDataType(name));

        for(int j = 0; j < tensorDims.nbDims; j++){
            totalTensorSize *= tensorDims.d[j];
        }

        void* devicePtr = nullptr;
        cudaMalloc(&devicePtr, totalTensorSize);
        m_DevicePtrMap[name] = devicePtr;
    }
}


InferencePipeline::InferencePipeline(const std::string& engineFilePath, Logger& useLogger):
    m_logger(useLogger),
    m_runtime(std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(m_logger))){

        // Check if runtime is set 
        if(!m_runtime){
            m_logger.log(ILogger::Severity::kERROR, "Failed to create TensorRT runtime.");
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        // Deserialize the model (in .engine file) and create an engine for running inference
        // Load the engine file onto a std::vector buffer

        std::vector<char> engineData = readEngineFileToArray(engineFilePath);

        // Deserialize the engineData to an Engine using the Runtime created
        m_engine = std::unique_ptr<ICudaEngine>(
            m_runtime->deserializeCudaEngine(engineData.data(), engineData.size())
            );
        
        if(!m_engine){   
            m_logger.log(ILogger::Severity::kERROR, "Failed to create TensorRT engine.");
            throw std::runtime_error("Failed to create TensorRT engine.");
        }

        allocateIODeviceMemory();
}   

// outputPtrs should be allocated outside this.
void InferencePipeline::runAsynchronousInference(const cv::Mat& preprocessedImage, std::unordered_map<std::string, void*>& outputPtrMap){

    std::string inputName;
    std::vector<std::string> outputNames;

    std::unique_ptr<IExecutionContext> context(m_engine->createExecutionContext());
    
    for(const auto& [name, d_ptr] : m_DevicePtrMap){
        if(m_engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT)
            inputName = name;
        else
            outputNames.emplace_back(name);
        
        bool assignFlag = context->setTensorAddress(name.c_str(), d_ptr);

        if(!assignFlag){
            throw std::runtime_error("Failed to set tensor Address for " + name + ".");
        }

    }

    size_t imageSize = preprocessedImage.total() * preprocessedImage.elemSize();
    cudaMemcpy(m_DevicePtrMap[inputName], preprocessedImage.data, imageSize, cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->enqueueV3(stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    for(int i = 0; i < outputNames.size(); i++){
        
        const std::string& name = outputNames[i];
        const Dims& outputShape = m_engine->getTensorShape(name.c_str());    
        
        size_t outputSize = getElementSize(m_engine->getTensorDataType(name.c_str()));

        for(int j = 0; j < outputShape.nbDims; j++){
            outputSize *= outputShape.d[j];
        }
        cudaMemcpy(outputPtrMap[name], m_DevicePtrMap[name], outputSize, cudaMemcpyDeviceToHost);
        cudaFree(m_DevicePtrMap[name]);
    }
}