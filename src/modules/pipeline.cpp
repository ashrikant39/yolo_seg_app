#include "pipeline.h"

std::vector<char> readEngineFileToArray(std::string fileName){

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


InferencePipeline::InferencePipeline(const std::string& engineFilePath, Logger& useLogger):
    m_logger(useLogger),
    m_runtime(std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(m_logger))){

        // Check if runtime is set 
        if(!m_runtime){
            m_logger.log(ILogger::Severity::kERROR, "Failed to create TensorRT runtime.");
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        //Create a new engine
        createNewEngine(engineFilePath);

}   



void InferencePipeline::createNewEngine(const std::string& engineFilePath){
    // Remove existing engine if any

    if(!m_runtime)
        throw std::runtime_error("No runtime. Cannot create a deserialized engine without a runtime");

    if(m_engine)
        m_engine.reset();

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
}


void InferencePipeline::LoadInputBuffer(const std::vector<cv::Mat>& batchInputs, void* gpuInputBuffer, int BATCH_SIZE, int HEIGHT, int WIDTH, int CHANNELS){
    size_t bufferSize = BATCH_SIZE * HEIGHT * WIDTH * CHANNELS * sizeof(float);
}

// void InferencePipeline::runAsynchronousInference(const std::vector<cv::Mat>& batchInputs){

// }

// void InferenceEngine::(const std::vector<cv::Mat>& batchInputs){
    
// }

