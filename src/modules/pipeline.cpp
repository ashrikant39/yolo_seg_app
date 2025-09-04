#include "pipeline.h"
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
#include <nvToolsExt.h>

#define NVTX_RANGE(name) do { nvtxRangePushA(name); } while (0)
#define NVTX_POP()      do { nvtxRangePop(); } while (0)


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

    nvinfer1::DataType tensorType = _engine->getTensorDataType(tensorName);
    const nvinfer1::Dims& tensorDims = _engine->getTensorShape(tensorName);

    size_t numElements = std::accumulate(
        tensorDims.d, 
        tensorDims.d + tensorDims.nbDims, 
        static_cast<size_t>(1),
        std::multiplies<>()
    );

    return numElements;
}


std::vector<std::string> InferencePipeline::getTensorNames(nvinfer1::TensorIOMode mode){
    
    int totalIOTensors = _engine->getNbIOTensors();
    std::vector<std::string> tensorNames;
    
    for(int i=0; i<totalIOTensors; i++){
        std::string name = _engine->getIOTensorName(i);

        if(_engine->getTensorIOMode(name.c_str()) == mode){
            tensorNames.push_back(std::move(name));
        }
    }
    
    return tensorNames;
}


bool InferencePipeline::createInferenceTensors(){

    for(int32_t i=0; i<_engine->getNbIOTensors(); i++){
        
        const char* name = _engine->getIOTensorName(i);

        _DeviceTensorMap[name].trtDtype = _engine->getTensorDataType(name);
        _DeviceTensorMap[name].numElements = getNumElements(name);
        _DeviceTensorMap[name].dims = _engine->getTensorShape(name);
        _DeviceTensorMap[name].mode = _engine->getTensorIOMode(name);
        
        size_t bytesPerElement = getElementSize(_DeviceTensorMap[name].trtDtype);

        cudaError_t error = cudaMalloc(
            &_DeviceTensorMap[name].ptr,
            _DeviceTensorMap[name].numElements * bytesPerElement
        );

        if(error != cudaSuccess){
            _logger.logConcatMessage(
                Severity::kINFO,
                "CudaMalloc Failed for {",
                name,
                "} : ",
                cudaGetErrorString(error),
                '\n'
            );
            return false;
        }

        if(_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT){

            _outputTensorMap[name] = {
                _DeviceTensorMap[name].trtDtype,
                _DeviceTensorMap[name].numElements,
                _DeviceTensorMap[name].dims,
                _DeviceTensorMap[name].mode,
                new cv::float16_t[_DeviceTensorMap[name].numElements]
            };
        }

    }

    return true;
}


void InferencePipeline::logModelInfo(){

    for(int idx=0; idx<_engine->getNbIOTensors(); idx++){

        const char* tensorName = _engine->getIOTensorName(idx);
        _logger.logTensorDims(
            Severity::kINFO,
            tensorName,
            _DeviceTensorMap[tensorName].dims
        );
    }

    _logger.log(Severity::kINFO, "Logging Layers Info of first and last few layers.\n");

    std::unique_ptr<nvinfer1::IEngineInspector> engineInspector(_engine->createEngineInspector());
    int numLayers = _engine->getNbLayers();
    
    for(int layerIdx=0; layerIdx<5; layerIdx++){
        _logger.logConcatMessage(
            Severity::kINFO,
            "Layer Index:",
            layerIdx,
            '\t',
            engineInspector->getLayerInformation(layerIdx, nvinfer1::LayerInformationFormat::kONELINE),
            '\n'
        );
        
    }

    for(int layerIdx=numLayers-5; layerIdx<numLayers; layerIdx++){
        _logger.logConcatMessage(
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
    _logger(Logger(logFilePath, Severity::kINFO)),
    _runtime(std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_logger))){

        // Check if runtime is set
        if(!_runtime){
            _logger.log(Severity::kERROR, "Failed to create TensorRT runtime.");
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }
        // Deserialize the model (in .engine file) and create an engine for running inference
        // Load the engine file onto a std::vector buffer

        std::vector<char> engineData = readEngineFileToArray(engineFilePath);

        // Deserialize the engineData to an Engine using the Runtime created
        _engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            _runtime->deserializeCudaEngine(engineData.data(), engineData.size())
            );
        
        if(!_engine){   
            _logger.log(Severity::kERROR, "Failed to create TensorRT engine.");
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
        _videoReader = std::make_unique<VideoFromDirectory>(
            videoDirPath,
            _DeviceTensorMap[inputName].dims.d[0],
            _DeviceTensorMap[inputName].dims.d[2],
            _DeviceTensorMap[inputName].dims.d[3],
            _logger
        );
        NVTX_POP();
        
        NVTX_RANGE("create_post_processor");
        _postProcessor = std::make_unique<PostProcessor>(_outputTensorMap, saveDirPath);
        NVTX_POP();
}

// DESTRUCTOR
InferencePipeline::~InferencePipeline(){

    for(auto& [name, d_tensor]: _DeviceTensorMap){
        cudaFree(d_tensor.ptr);
 
        if(_outputTensorMap.find(name) != _outputTensorMap.end()){
            delete[] _outputTensorMap[name].ptr;
        }   
    }

}

bool InferencePipeline::runAsynchronousInference(const cv::Mat& preprocessedImages){

    std::string inputName = getTensorNames(nvinfer1::TensorIOMode::kINPUT)[0];
    std::vector<std::string> outputNames = getTensorNames(nvinfer1::TensorIOMode::kOUTPUT);

    std::unique_ptr<nvinfer1::IExecutionContext> context(_engine->createExecutionContext());
    
    NVTX_RANGE("AddressSetting");
    for(const auto& [name, d_tensor] : _DeviceTensorMap){
        
        bool assignFlag = context->setTensorAddress(name.c_str(), d_tensor.ptr);

        if(!assignFlag){
            std::cerr << "Failed to set tensor Address for {" << name << "}.";
            return false; 
        }
    }
    NVTX_POP();
    
    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);

    if(error != cudaSuccess){
        std::cerr << "cudaStreamCreate Failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    size_t bytesPerElement = getElementSize(_DeviceTensorMap[inputName].trtDtype);

    NVTX_RANGE("HostToDevice");
    error = cudaMemcpyAsync(
        _DeviceTensorMap[inputName].ptr, 
        preprocessedImages.ptr<cv::float16_t>(0), 
        _DeviceTensorMap[inputName].numElements * bytesPerElement, 
        cudaMemcpyHostToDevice, 
        stream
    );
    NVTX_POP();

    if(error != cudaSuccess){
        std::cerr << "MemCpuAsync Failed for {" << inputName << "} : " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    NVTX_RANGE("EnqueueV3");
    if(!context->enqueueV3(stream)){
        std::cerr << "EnqueueV3 Failed: ";
        _logger.log(Severity::kERROR, "EnqueueV3 Failed.");
        return false;
    }
    NVTX_POP();
    
    NVTX_RANGE("DeviceToHost");
    for(int i=0; i<outputNames.size(); i++){
        
        const std::string& name = outputNames[i];
                
        error = cudaMemcpyAsync(
            _outputTensorMap[name].ptr,
            _DeviceTensorMap[name].ptr,
            _outputTensorMap[name].numElements * bytesPerElement, 
            cudaMemcpyDeviceToHost,
            stream
        );

        if(error != cudaSuccess){
            std::cerr << "MemCpuAsync Failed for {" << name << "} : " << cudaGetErrorString(error) << std::endl;
        }
    }
    NVTX_POP();

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

    size_t totalBatches = _videoReader->getTotalBatches();
    size_t totalImgs = _videoReader->getTotalImages();

    auto beginCompute = std::chrono::high_resolution_clock::now();

    for(size_t batchIdx=0; batchIdx < totalBatches; batchIdx++){

        NVTX_RANGE("PreProcessBatch");
        const auto& [imageData, imagePaths] = 
                _videoReader->getBatchDataPreProcessed(
                    batchIdx,
                    _logger,
                    VideoOptions::NORM_FACTOR_ADD_TO_SCALED,
                    VideoOptions::NORM_FACTOR_SCALING_MUL
                );            
        NVTX_POP();
        
        NVTX_RANGE("Inference");
        if(!runAsynchronousInference(imageData)){
            std::cerr << "Inference Failed" << std::endl;
        }
        NVTX_POP();

        NVTX_RANGE("PostProcess");
        _postProcessor->postProcessOutputs(_outputTensorMap, imagePaths, _logger);
        NVTX_POP();
    }

    auto endCompute = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(endCompute - beginCompute).count();
        
    _logger.logConcatMessage(
        Severity::kINFO,
        "Total Compute Time: ",
        duration,
        '\n'
    );

    _logger.logConcatMessage(
        Severity::kINFO,
        "FPS: ",
        static_cast<double>(totalImgs) / duration,
        '\n'
    );
}
