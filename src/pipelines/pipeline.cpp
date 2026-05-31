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
#include <assert.h>

#include "pipeline/pipeline.hpp"
#include "post_process/factory/PostProcessFactory.hpp"
#include "post_process/utils/PostProcessUtils.hpp"

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
        size_t{1},
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

void InferencePipeline::allocatePostprocTensors() {

    NVTX_RANGE("CreateInferTensors");
    if (m_postProcessTensorMap.empty()) {

        for (const auto& [name, tensor] : m_HostTensorMap) {

            if (tensor.getIOMode() != nvinfer1::TensorIOMode::kOUTPUT) {
                continue;
            }            

            m_postProcessTensorMap.emplace(
                name,
                Tensor<UniquePtrToArray>(
                    nvinfer1::DataType::kFLOAT,
                    tensor.getDims(),
                    nvinfer1::TensorIOMode::kOUTPUT)
            );
        }
    }
    NVTX_POP();
}

// CONSTRUCTOR
InferencePipeline::InferencePipeline(
    const fs::path& engineFilePath, 
    const fs::path& logFilePath,
    const fs::path& videoDirPath,
    const fs::path& saveDirPath,
    bool logModelInformation):
    m_logger(logFilePath, Severity::kINFO),
    m_runtime(std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger))){

        m_logger.logConcatMessage(Severity::kINFO, "OpenCV threads: ", cv::getNumThreads(), '\n');
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
            m_runtime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()
            )
        );
        
        if(!m_engine){   
            m_logger.log(Severity::kERROR, "Failed to create TensorRT engine.");
            throw std::runtime_error("Failed to create TensorRT engine.");
        }
        
        NVTX_RANGE("ALLOCATE_TENSORS");
        allocateTensorMemoryTRT(m_HostTensorMap);
        allocateTensorMemoryTRT(m_DeviceTensorMap);
        allocatePostprocTensors();
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
            m_HostTensorMap[inputName].ptr<cv::float16_t>() // I have currently hardcoded to use float16 input
        );
        NVTX_POP();
        
        NVTX_RANGE("create_post_processor");
        PostProcessorConfig config;
        m_postProcessor = createPostProcessor(config);
        NVTX_POP();
        
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

        NVTX_RANGE("AddressSetting");
        for(auto& [name, d_tensor] : m_DeviceTensorMap){
            if(!m_context->setTensorAddress(name.c_str(), d_tensor.rawPtr())){
                std::cerr << "Failed to set tensor Address for {" << name << "}.";
            }
        }
        NVTX_POP();
        nvtxNameCuStreamA(m_stream.get(), "InferenceStream");
}


void InferencePipeline::runInference(){

    NVTX_RANGE("SetupInference");
    TensorViewMap deviceTensors = getTensorViewMap(m_DeviceTensorMap);
    TensorViewMap hostTensors = getTensorViewMap(m_HostTensorMap);

    std::string inputName = getTensorNames(nvinfer1::TensorIOMode::kINPUT)[0];
    std::vector<std::string> outputNames = getTensorNames(nvinfer1::TensorIOMode::kOUTPUT);

    NVTX_POP();
    // error = cudaMemPrefetchAsync(
    //     m_DeviceTensorMap[inputName].rawPtr(),
    //     bytesPerElement * numInputElements,
    //     {cudaMemLocationType::cudaMemLocationTypeDevice, 0},
    //     0,
    //     stream
    // );
    cudaError_t error;

    NVTX_RANGE("HostToDeviceTransfer_1");
    error = cudaMemcpyAsync(
        deviceTensors[inputName].data,
        hostTensors[inputName].data,
        hostTensors[inputName].totalBytes,
        cudaMemcpyHostToDevice,
        m_stream.get()
    );
    NVTX_POP();

    if(error != cudaSuccess){
        throw std::runtime_error("cudaMemPrefetchAsync from cpu to gpu Failed: " + std::string(cudaGetErrorString(error)) + '\n');
    }

    NVTX_RANGE("EnqueueV3");
    if ( !m_context->enqueueV3(m_stream.get()) ) {
        m_logger.log(Severity::kERROR, "EnqueueV3 Failed.");
        throw std::runtime_error("EnqueueV3 Failed: \n");
    }
    NVTX_POP();

    NVTX_RANGE("CompleteDeviceToHostTransfer");
    for (const auto& name : outputNames) {

        // size_t bytesPerElement = getElementSize(m_DeviceTensorMap[name].getDtype());
        // size_t numElements = m_DeviceTensorMap[name].getNumElements();

        // error = cudaMemPrefetchAsync(
        //     m_DeviceTensorMap[name].rawPtr(),
        //     bytesPerElement * numElements,
        //     {cudaMemLocationType::cudaMemLocationTypeHost, 0},
        //     0,
        //     stream
        // );
        NVTX_RANGE("DeviceToHostTransfer_1");
        error = cudaMemcpyAsync(
            hostTensors[name].data,
            deviceTensors[name].data,
            hostTensors[name].totalBytes,
            cudaMemcpyDeviceToHost,
            m_stream.get()
        );
        
        if(error != cudaSuccess){
            throw std::runtime_error(
                "cudaMemPrefetchAsync from cpu to gpu Failed for : " + 
                std::string(name) + 
                ' ' + 
                std::string(cudaGetErrorString(error)) + 
                '\n'
            );
        }
        NVTX_POP();
    }
    NVTX_POP();

    NVTX_RANGE("StreamSync2");
    error = cudaStreamSynchronize(m_stream.get());
    
    if(error != cudaSuccess){
        std::cerr << "Stream Synchronization Failed." << std::endl;
        throw std::runtime_error("Stream Synchronization Failed After CPU->GPU." + std::string(cudaGetErrorString(error)) + '\n');
    }
    NVTX_POP();

}

void InferencePipeline::copyAndCastForPostProcessing() {

    for (auto& [m_name, m_tensor] : m_postProcessTensorMap) {
        
        const auto& outputTensor = m_HostTensorMap[m_name];

        if (outputTensor.getDtype() == nvinfer1::DataType::kHALF){            
            castHalfToFloat(m_tensor.ptr<float>(), outputTensor.ptr<__half>(), outputTensor.getNumElements());
        }

        else if (outputTensor.getDtype() == nvinfer1::DataType::kFLOAT) {
            std::memcpy(m_tensor.ptr<float>(), outputTensor.ptr<float>(), outputTensor.getNumElements() * sizeof(float));
        }
        else {
            throw std::runtime_error("GPU Output : Wrong type");
        }
        
    }
}


void InferencePipeline::runInferencePipeline(bool saveDetsAsFile, bool drawMasksOnImage){

    size_t totalBatches = m_batchLoader->getTotalBatches();
    size_t totalImgs = m_batchLoader->getTotalImages();
    const size_t batchSize = m_batchLoader->getBatchSize();
    const auto& allPaths = m_batchLoader->getFileNames();

    auto beginCompute = std::chrono::high_resolution_clock::now();

    for(size_t batchIdx=0; batchIdx < totalBatches; batchIdx++) {

        const size_t startIdx = batchIdx * batchSize;
        const size_t count = std::min(batchSize, totalImgs - startIdx);
        std::vector<fs::path> batchPaths(
            allPaths.begin() + static_cast<std::ptrdiff_t>(startIdx),
            allPaths.begin() + static_cast<std::ptrdiff_t>(startIdx + count));

        try {

            m_logger.log(Severity::kINFO, "Batch Loading");
            NVTX_RANGE("PreProcessBatch");
            m_batchLoader->loadBatchDataPreProcessed(
                batchIdx,
                m_logger,
                VideoOptions::NORM_FACTOR_ADD_TO_SCALED,
                VideoOptions::NORM_FACTOR_SCALING_MUL
            );
            NVTX_POP();
            
            NVTX_RANGE("Inference");
            runInference();
            NVTX_POP();

            NVTX_RANGE("CopyAndCastToCpu");
            copyAndCastForPostProcessing();
            NVTX_POP();

            NVTX_RANGE("PostProcess");
            TensorViewMap postprocViews = getTensorViewMap(m_postProcessTensorMap);

            std::vector<PostProcessOutput> outputs;
            m_postProcessor->process(postprocViews, outputs, m_logger, m_stream.get());
            NVTX_POP();

        } catch (const std::exception& e) { 
            m_logger.logConcatMessage(
                Severity::kERROR,
                "Batch ",
                batchIdx,
                " failed: ",
                e.what(),
                '\n'
            );
            continue;
        }
    }

    auto endCompute = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(endCompute - beginCompute).count();
        
    m_logger.logConcatMessage(
        Severity::kINFO,
        "Total Compute Time: ",
        duration,
        "s\n"
    );

    m_logger.logConcatMessage(
        Severity::kINFO,
        "FPS: ",
        static_cast<double>(totalImgs) / duration,
        '\n'
    );
}
