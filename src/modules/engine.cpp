#include "engine.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <cuda_runtime.h>
#include <opencv2/dnn/dnn.hpp>
#include <cstring>

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

TensorSizeInfo InferenceEngine::getTensorTotalElemsAndBytes(const char* tensorName){

    nvinfer1::DataType tensorType = m_engine->getTensorDataType(tensorName);
    const nvinfer1::Dims& tensorDims = m_engine->getTensorShape(tensorName);

    size_t el = getElementSize(tensorType);

    size_t totalTensorSize = std::accumulate(
        tensorDims.d, 
        tensorDims.d + tensorDims.nbDims, 
        static_cast<size_t>(1),
        std::multiplies<>()
    );

    return {totalTensorSize, totalTensorSize * el};
}


bool InferenceEngine::allocateIOMemory(){

    for(int32_t i=0; i<m_engine->getNbIOTensors(); i++){
        
        const char* name = m_engine->getIOTensorName(i);

        auto [totalElements, totalBytes] = getTensorTotalElemsAndBytes(name);
        cudaError_t error = cudaMalloc(&m_DevicePtrMap[name], totalBytes);

        if(error != cudaSuccess){
            std::stringstream msg;
            msg << "CudaMalloc Failed for {" << name << "} : " << cudaGetErrorString(error);
            std::cerr << msg.str();
            m_logger.log(Severity::kERROR, msg.str().c_str());
            return false;
        }

        if(m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT){
            m_outputPtrMap[name] = new cv::float16_t[totalElements];
            m_postProcessPtrMap[name] = new float[totalElements];
        }
    }

    return true;
}


void InferenceEngine::logModelInfo(){

    for(int idx=0; idx<m_engine->getNbIOTensors(); idx++){
        const char* tensorName = m_engine->getIOTensorName(idx);
        m_logger.logTensorDims(
            Severity::kINFO,
            tensorName,
            m_engine->getTensorShape(tensorName)
        );
    }

    std::stringstream modelInfo;
    modelInfo << "Logging Layers Info of first and last few layers.\n";

    std::unique_ptr<nvinfer1::IEngineInspector> engineInspector(m_engine->createEngineInspector());
    int numLayers = m_engine->getNbLayers();
    
    for(int layerIdx=0; layerIdx<5; layerIdx++){
        modelInfo << "Layer Index:" << layerIdx << "\t" << engineInspector->getLayerInformation(layerIdx, nvinfer1::LayerInformationFormat::kONELINE) << '\n';
        
    }

    for(int layerIdx=numLayers-5; layerIdx<numLayers; layerIdx++){
        modelInfo << "Layer Index:" << layerIdx << "\t" << engineInspector->getLayerInformation(layerIdx, nvinfer1::LayerInformationFormat::kONELINE) << '\n';
    }

    m_logger.log(Severity::kINFO, modelInfo.str().c_str());

}

// CONSTRUCTOR
InferenceEngine::InferenceEngine(const fs::path& engineFilePath, Logger& useLogger, bool logModelInformation):
    m_logger(useLogger),
    m_runtime(std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(useLogger))){

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

        if(!allocateIOMemory()){
            throw std::runtime_error("Failed to allocate IO memory");
        }

        if(logModelInformation){
            logModelInfo();
        }
}

// DESTRUCTOR
InferenceEngine::~InferenceEngine(){

    for(auto& [name, ptr]: m_DevicePtrMap){
        cudaFree(ptr);
 
        if(m_outputPtrMap.find(name) != m_outputPtrMap.end()){
            delete[] m_outputPtrMap[name];
            delete[] m_postProcessPtrMap[name];
        }   
    }

}

bool InferenceEngine::runAsynchronousInference(const std::vector<cv::float16_t>& preprocessedImageBuffer){

    std::string inputName = getTensorNames(nvinfer1::TensorIOMode::kINPUT)[0];
    std::vector<std::string> outputNames = getTensorNames(nvinfer1::TensorIOMode::kOUTPUT);

    std::unique_ptr<nvinfer1::IExecutionContext> context(m_engine->createExecutionContext());
    
    for(const auto& [name, d_ptr] : m_DevicePtrMap){
        
        bool assignFlag = context->setTensorAddress(name.c_str(), d_ptr);

        if(!assignFlag){
            std::cerr << "Failed to set tensor Address for {" << name << "}.";
            return false; 
        }
    }
                        // #float16 values          
    size_t imageSize = preprocessedImageBuffer.size() * sizeof(cv::float16_t);
    
    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);

    if(error != cudaSuccess){
        std::cerr << "cudaStreamCreate Failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    error = cudaMemcpyAsync(
        m_DevicePtrMap[inputName], 
        preprocessedImageBuffer.data(), 
        imageSize, 
        cudaMemcpyHostToDevice, 
        stream
    );

    if(error != cudaSuccess){
        std::cerr << "MemCpuAsync Failed for {" << inputName << "} : " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    if(!context->enqueueV3(stream)){
        std::cerr << "EnqueueV3 Failed: ";
        m_logger.log(Severity::kERROR, "EnqueueV3 Failed.");
        return false;
    }
    
    std::vector<size_t> outputElems;

    for(int i=0; i<outputNames.size(); i++){
        
        const std::string& name = outputNames[i];
        auto [totalOutpuElems, totalOutputBytes] = getTensorTotalElemsAndBytes(name.c_str());
        
        outputElems.push_back(totalOutpuElems);
        
        error = cudaMemcpyAsync(
            m_outputPtrMap[name], 
            m_DevicePtrMap[name], 
            totalOutputBytes, 
            cudaMemcpyDeviceToHost, 
            stream
        );

        if(error != cudaSuccess){
            std::cerr << "MemCpuAsync Failed for {" << name << "} : " << cudaGetErrorString(error) << std::endl;
        }
    }

    error = cudaStreamSynchronize(stream);
    
    for(int i=0; i<outputNames.size(); i++){

        const std::string& name = outputNames[i];
        cv::float16_t *h_ptr = m_outputPtrMap[name];
        size_t total = outputElems[i];
        float sum = 0.0f;

        for(int j=0; j<total; j++){
            sum += (h_ptr[j]/total);
        }

        std::cout << "Total: " << sum << "\n";
    }


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


std::vector<std::string> InferenceEngine::getTensorNames(nvinfer1::TensorIOMode mode){
    
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


std::vector<nvinfer1::Dims> InferenceEngine::getTensorShapes(nvinfer1::TensorIOMode mode){

    std::vector<nvinfer1::Dims> shapes;
    
    for(std::string& tensorName : getTensorNames(mode)){
        shapes.push_back(m_engine->getTensorShape(tensorName.c_str()));
    }

    return shapes;

}

std::vector<nvinfer1::DataType> InferenceEngine::getTensorDataTypes(nvinfer1::TensorIOMode mode){
    std::vector<nvinfer1::DataType> types;

    for(std::string& tensorName : getTensorNames(mode)){
        types.push_back(m_engine->getTensorDataType(tensorName.c_str()));
    }

    return types;
}

nvinfer1::Dims InferenceEngine::getInputDims(){ 
    return getTensorShapes(nvinfer1::TensorIOMode::kINPUT)[0];
}

// auto as a return type cannot make a function return multiple types through multiple paths
// Eigen::Tensor should get its rank as a compile time const.

// So, all CPU computations are to be done in float32
// Also, the rank of the tensor is to be kept 4.
// [[NOTE]]:  You have to use a dimension array of type Eigen::DSizes<Eigen::Index, 4> to use an array as the dims of an eigen tensor.
// TensorMap does not have a default constructor; so we can return a view tensor based on name instead of a view tensor map.


TensorViewSharedPtr<float, 4> InferenceEngine::getOutputTensorViewFromMemory4D(const std::string& outputName){

    if(!m_outputPtrMap[outputName]){
        throw std::invalid_argument("Cannot create 4D tensor from null pointer");
    }
    
    nvinfer1::Dims tensorDims = m_engine->getTensorShape(outputName.c_str());
    auto [totalElems, totalBytes] = getTensorTotalElemsAndBytes(outputName.c_str());
    Eigen::DSizes<Eigen::Index, 4> eigenTensorDims;

    for(int i=0; i<4; i++){
        eigenTensorDims[i] = static_cast<Eigen::Index>(tensorDims.d[i] >  0 ? tensorDims.d[i] : 1);
    }

    copyDataToFloat32(
        m_outputPtrMap[outputName],
        m_postProcessPtrMap[outputName],
        totalElems,
        m_logger
    );

    return std::make_shared<TensorView<float, 4>>(m_postProcessPtrMap[outputName], eigenTensorDims);

}   


TensorViewSharedPtrMap<float, 4> InferenceEngine::getOutputTensorViewMap(){

    TensorViewSharedPtrMap<float, 4> outputTensorViewMap;

    for(auto& name: getTensorNames(nvinfer1::TensorIOMode::kOUTPUT)){
        outputTensorViewMap.emplace(name, getOutputTensorViewFromMemory4D(name));
    }

    return outputTensorViewMap;
}


void copyDataToFloat32(
    cv::float16_t* sourceDataPtr, 
    float* destDataPtr, 
    size_t totalElements, 
    Logger& logger){ 

        assert(reinterpret_cast<uintptr_t>(sourceDataPtr) % alignof(Eigen::half) == 0); // check given by GPT

        TensorView<Eigen::half, 1> sourceData(
            reinterpret_cast<Eigen::half*>(sourceDataPtr),
            totalElements
        );

        TensorView<float, 1> copiedData(
            destDataPtr, 
            totalElements
        );

        for(int i=0; i<totalElements; i++){
            copiedData(i) = static_cast<float>(sourceData(i));
        }
}


// bboxes -> const std::vector<cv::Rect>&
// scores -> const std::vector<float>&
// score_threshold (same as conf_thres)
// nms_threshold (same as iou_thres)
// eta, top_k

// std::vector<std::vector<int>> applyNonMaximalSuppressionOnBatch(){

// }
void InferenceEngine::postProcessOutputs(const fs::path& saveDir, const std::vector<fs::path>& imagePaths){
    /*
        name: images
            tensor: float16[16,3,512,1024]
            
            name: output0
            tensor: float16[16,300,38]
            
            name: output1
            tensor: float16[16,32,128,256]

            prototype masks -> output1
            mask coeffs + boxes -> output0
            In output0,
                mask coeffs         -> output0[...,6:]
                boxes               -> output0[..., :4]
                objectness score    -> output0[...,4]
                cls score           -> output0[...,5] (Since only one class)
    */

    TensorViewSharedPtrMap<float, 4> outputTensors = getOutputTensorViewMap();
    TensorViewSharedPtr<float, 4> outputCoeffs = outputTensors["output0"];

    auto dims0 = outputCoeffs->dimensions();    //  (16, 300, 38, 1) 
    int batchSize = dims0[0];
    int Nboxes = dims0[1];
    int NCoeffs = dims0[2];

    TensorViewSharedPtr<float, 4> prototypeMasks = outputTensors["output1"];
    auto dims1 = prototypeMasks->dimensions();  //  (16, 32, 512, 512)
    int featDim = dims1[1];
    int maskH = dims1[2];
    int maskW = dims1[3];

    auto batchBoxes = outputCoeffs->slice(
        Eigen::DSizes<Eigen::Index, 4>{0, 0, 0, 0},
        Eigen::DSizes<Eigen::Index, 4>{batchSize, Nboxes, 4, 1}
    ).chip(0, 3);    // (16, 300, 4, 1) -> (16, 300, 4)

    auto batchScores = outputCoeffs->slice(
        Eigen::DSizes<Eigen::Index, 4>{0, 0, 4, 0},
        Eigen::DSizes<Eigen::Index, 4>{batchSize, Nboxes, 1, 1}
    ).chip(0, 3).chip(0, 2);    // (16, 300, 1, 1) -> (16, 300)
}
