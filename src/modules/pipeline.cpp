#include "pipeline.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include <stdexcept>
#include <numeric>
#include <cuda_runtime.h>
#include <opencv2/dnn/dnn.hpp>
#include <cstring>
#include <chrono>
#include <nvtx3/nvToolsExt.h>
#include <assert.h>

#ifndef NDEBUG
    #define NVTX_RANGE(name) do { nvtxRangePushA(name); } while (0)
    #define NVTX_POP()      do { nvtxRangePop(); } while (0)
#else
    #define NVTX_RANGE(name) do { } while(0)
    #define NVTX_POP()      do { } while(0)
#endif


size_t getElementSize(nvinfer1::DataType dtype){

    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: 
            return 4;
        case nvinfer1::DataType::kHALF: 
            return 2;
        case nvinfer1::DataType::kBF16: 
            return 2;
        case nvinfer1::DataType::kINT8: 
            return 1;
        case nvinfer1::DataType::kINT32: 
            return 4;
        default: throw std::runtime_error("Unknown data type");
    }
}

std::vector<char> readEngineFileToArray(const fs::path& fileName){

    std::ifstream file(fileName, std::ios::binary);
    
    if(!file.is_open()){
        throw std::runtime_error("Failed to open engine file: " + fileName.string());
    }

    size_t size = fs::file_size(fileName);
    std::vector<char> engineData(size);

    if(!file.read(engineData.data(), size)){
        throw std::runtime_error("Failed to read engine file: " + fileName.string());
    }

    return engineData;

}

size_t InferencePipeline::getNumElements(const char* tensorName){

    nvinfer1::DataType tensorType = m_engine->getTensorDataType(tensorName);
    const nvinfer1::Dims& tensorDims = m_engine->getTensorShape(tensorName);

    size_t numElements = std::accumulate(
        tensorDims.d, 
        tensorDims.d + tensorDims.nbDims, 
        static_cast<size_t>(1),
        std::multiplies<>()
    );

    return numElements;
}


std::vector<std::string> InferencePipeline::getTensorNames(nvinfer1::TensorIOMode mode){
    
    int totalIOTensors = m_engine->getNbIOTensors();
    std::vector<std::string> tensorNames;
    
    for(int i=0; i<totalIOTensors; i++){
        std::string name = m_engine->getIOTensorName(i);

        if(m_engine->getTensorIOMode(name.c_str()) == mode){
            tensorNames.push_back(std::move(name));
        }
    }
    
    return tensorNames;
}


bool InferencePipeline::createInferenceTensors(){

    for(int32_t i=0; i<m_engine->getNbIOTensors(); i++){
        
        const char* name = m_engine->getIOTensorName(i);

        m_DeviceTensorMap[name] = {
            m_engine->getTensorDataType(name),
            m_engine->getTensorShape(name),
            m_engine->getTensorIOMode(name)
        };

    }

    return true;
}


void InferencePipeline::logModelInfo(){

    for(int idx=0; idx<m_engine->getNbIOTensors(); idx++){

        const char* tensorName = m_engine->getIOTensorName(idx);
        m_logger.logTensorDims(
            Severity::kINFO,
            tensorName,
            m_DeviceTensorMap[tensorName].getDims()
        );
    }

    m_logger.log(Severity::kINFO, "Logging Layers Info of first and last few layers.\n");

    std::unique_ptr<nvinfer1::IEngineInspector> engineInspector(m_engine->createEngineInspector());
    int numLayers = m_engine->getNbLayers();
    
    for(int layerIdx=0; layerIdx<5; layerIdx++){
        m_logger.logConcatMessage(
            Severity::kINFO,
            "Layer Index:",
            layerIdx,
            '\t',
            engineInspector->getLayerInformation(layerIdx, nvinfer1::LayerInformationFormat::kONELINE),
            '\n'
        );
        
    }

    for(int layerIdx=numLayers-5; layerIdx<numLayers; layerIdx++){
        m_logger.logConcatMessage(
            Severity::kINFO,
            "Layer Index:",
            layerIdx,
            '\t',
            engineInspector->getLayerInformation(layerIdx, nvinfer1::LayerInformationFormat::kONELINE),
            '\n'
        );
    }

}

// CONSTRUCTOR
InferencePipeline::InferencePipeline(
    const fs::path& engineFilePath, 
    const fs::path& logFilePath,
    const fs::path& videoDirPath,
    const fs::path& saveDirPath,
    bool logModelInformation
):
    m_logger(logFilePath, Severity::kINFO),
    m_runtime(std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger))){

        // Check if runtime is set
        if(!m_runtime){
            m_logger.log(Severity::kERROR, "Failed to create TensorRT runtime.");
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }
        // Deserialize the model (in .engine file) and create an engine for running inference
        // Load the engine file onto a std::vector buffer

        std::vector<char> engineData = readEngineFileToArray(engineFilePath);

        // Deserialize the engineData to an Engine using the Runtime created
        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            m_runtime->deserializeCudaEngine(engineData.data(), engineData.size())
            );
        
        if(!m_engine){   
            m_logger.log(Severity::kERROR, "Failed to create TensorRT engine.");
            throw std::runtime_error("Failed to create TensorRT engine.");
        }
        
        NVTX_RANGE("create_tensors");
        if(!createInferenceTensors()){
            throw std::runtime_error("Failed to allocate IO memory");
        }
        NVTX_POP();

        if(logModelInformation){
            logModelInfo();
        }

        std::string inputName = getTensorNames(nvinfer1::TensorIOMode::kINPUT)[0];
        
        NVTX_RANGE("create_video");
        nvinfer1::Dims inputDims = m_DeviceTensorMap[inputName].getDims();

        m_batchLoader = std::make_unique<ImageBatchLoader>(
            videoDirPath,
            inputDims.d[0],
            inputDims.d[2],
            inputDims.d[3],
            m_logger,
            m_DeviceTensorMap[inputName].ptr()
        );
        NVTX_POP();
        
        NVTX_RANGE("create_post_processor");
        m_postProcessor = std::make_unique<PostProcessor>(
            saveDirPath,
            inputDims.d[3],
            inputDims.d[2]);
        NVTX_POP();
        
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

        NVTX_RANGE("AddressSetting");
        for(auto& [name, d_tensor] : m_DeviceTensorMap){
            if(!m_context->setTensorAddress(name.c_str(), d_tensor.ptr())){
                std::cerr << "Failed to set tensor Address for {" << name << "}.";
            }
        }
        NVTX_POP();
}


bool InferencePipeline::runInference(){

    std::string inputName = getTensorNames(nvinfer1::TensorIOMode::kINPUT)[0];
    std::vector<std::string> outputNames = getTensorNames(nvinfer1::TensorIOMode::kOUTPUT);

    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);

    if(error != cudaSuccess){
        std::cerr << "cudaStreamCreate Failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    size_t bytesPerElement = getElementSize(m_DeviceTensorMap[inputName].getDtype());
    size_t numInputElements = m_DeviceTensorMap[inputName].getNumElements();

    cudaMemPrefetchAsync(
        m_DeviceTensorMap[inputName].ptr(),
        bytesPerElement * numInputElements,
        {cudaMemLocationType::cudaMemLocationTypeDevice, 0},
        0,
        stream
    );

    NVTX_RANGE("EnqueueV3");
    if(!m_context->enqueueV3(stream)){
        std::cerr << "EnqueueV3 Failed: ";
        m_logger.log(Severity::kERROR, "EnqueueV3 Failed.");
        return false;
    }
    NVTX_POP();

    error = cudaStreamSynchronize(stream);
    if(error != cudaSuccess){
        std::cerr << "Stream Synchronization Failed." << std::endl;
        return false;
    }

    cudaMemPrefetchAsync(
        m_DeviceTensorMap[inputName].ptr(),
        bytesPerElement * numInputElements,
        {cudaMemLocationType::cudaMemLocationTypeHost, 0},
        0,
        stream);

    error = cudaStreamSynchronize(stream);
    if(error != cudaSuccess){
        std::cerr << "Stream Synchronization Failed." << std::endl;
        return false;
    }

    error = cudaStreamDestroy(stream);
    
    if(error != cudaSuccess){
        std::cerr << "Stream Destruction Failed." << std::endl;
        return false;
    }

    return true;
}


void InferencePipeline::runInferencePipeline(){

    size_t totalBatches = m_batchLoader->getTotalBatches();
    size_t totalImgs = m_batchLoader->getTotalImages();
    const size_t batchSize = m_batchLoader->getBatchSize();
    const auto& allPaths = m_batchLoader->getFileNames();

    auto beginCompute = std::chrono::high_resolution_clock::now();

    for(size_t batchIdx=0; batchIdx < totalBatches; batchIdx++){

        const size_t startIdx = batchIdx * batchSize;
        const size_t count = std::min(batchSize, totalImgs - startIdx);
        std::vector<fs::path> batchPaths(
            allPaths.begin() + static_cast<std::ptrdiff_t>(startIdx),
            allPaths.begin() + static_cast<std::ptrdiff_t>(startIdx + count));

        NVTX_RANGE("PreProcessBatch");

        m_batchLoader->loadBatchDataPreProcessed(
            batchIdx,
            m_logger,
            VideoOptions::NORM_FACTOR_ADD_TO_SCALED,
            VideoOptions::NORM_FACTOR_SCALING_MUL);            
        NVTX_POP();
        
        NVTX_RANGE("Inference");
        if(!runInference()){
            std::cerr << "Inference Failed" << std::endl;
        }
        NVTX_POP();

        NVTX_RANGE("PostProcess");
        m_postProcessor->postProcessOutputs(m_DeviceTensorMap, batchPaths, m_logger);
        NVTX_POP();
    }

    auto endCompute = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(endCompute - beginCompute).count();
        
    m_logger.logConcatMessage(
        Severity::kINFO,
        "Total Compute Time: ",
        duration,
        '\n'
    );

    m_logger.logConcatMessage(
        Severity::kINFO,
        "FPS: ",
        static_cast<double>(totalImgs) / duration,
        '\n'
    );
}
