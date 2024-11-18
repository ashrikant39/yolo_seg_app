#include "engine.h"


InferenceEngine::InferenceEngine(const std::string& engineFilePath, Logger& useLogger):
    m_logger(useLogger),
    m_runtime(
        std::unique_ptr<NVRuntime>(nvinfer1::createInferRuntime(m_logger))
        )
{
    // Check if runtime is set 
    if(!m_runtime)
    {
        m_logger.log(NVLogger::Severity::kERROR, "Failed to create TensorRT runtime.");
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    //Create a new engine
    createNewEngine(engineFilePath);

    //Create a new execution context
    createNewExecutionContext();

}   



void InferenceEngine::createNewEngine(const std::string& engineFilePath)
{
    // Remove existing engine if any

    if(!m_runtime)
        throw std::runtime_error("No runtime. Cannot create a deserialized engine without a runtime");

    if(m_engine)
        m_engine.reset();

    // Deserialize the model (in .engine file) and create an engine for running inference
    // Load the engine file onto a std::vector buffer

    std::ifstream engineFileStream(engineFilePath, std::ios::binary);

    if(!engineFileStream)
    {
        m_logger.log(NVLogger::Severity::kERROR, "Failed to open engine file.");
        throw std::runtime_error("Failed to open engine file.");
    }

    // Once a file pointer is created, compute the size of the engine file
    // to create a buffer of that size

    engineFileStream.seekg(0, std::ios::end);
    size_t engineFileSize = static_cast<size_t>(engineFileStream.tellg());
    engineFileStream.seekg(0, std::ios::beg);
    // Use an std::vector to store the contents of the engine file
    // as a vector of chars  
    // Use a vector, as the size of an std::array has to be known at compile time.
    // NOTE: read function expects a char* pointer

    std::vector<char> engineData(engineFileSize);
    engineFileStream.read(engineData.data(), engineFileSize);
    engineFileStream.close();

    // Deserialize the engineData to an Engine using the Runtime created
    m_engine = std::unique_ptr<NVEngine>(
        m_runtime->deserializeCudaEngine(engineData.data(), engineFileSize)
        );
    
    if(!m_engine) 
    {   
        m_logger.log(NVLogger::Severity::kERROR, "Failed to create TensorRT engine.");
        throw std::runtime_error("Failed to create TensorRT engine.");
    }
}


void InferenceEngine::createNewExecutionContext()
{
    if(!m_engine)
        throw std::runtime_error("No engine. Cannot create a new execution context without an engine.");

    // Create a new execution context
    // Execution context uses the input, output buffers on GPU to running the inference

    m_executionContext = std::unique_ptr<NVExecContext>(m_engine->createExecutionContext());

    if (!m_executionContext) 
    {
        m_logger.log(NVLogger::Severity::kERROR, "Failed to create TensorRT execution context.");
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }   
}

void InferenceEngine::LoadInputBuffer(const std::vector<cv::Mat>& batchInputs, void* gpuInputBuffer, int BATCH_SIZE, int HEIGHT, int WIDTH, int CHANNELS)
{
    size_t bufferSize = BATCH_SIZE * HEIGHT * WIDTH * CHANNELS * sizeof(float);
}

void InferenceEngine::runAsynchronousInference(const std::vector<cv::Mat>& batchInputs)



