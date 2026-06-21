#include "backends/utils/trtUtils.hpp"

std::vector<char> readEngineFileToArray(const fs::path& fileName) {

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


std::vector<std::string> getTensorNames(
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
    nvinfer1::TensorIOMode mode
) {
    
    int totalIOTensors = engine->getNbIOTensors();
    std::vector<std::string> tensorNames;
    tensorNames.reserve(totalIOTensors);
    
    for(int i=0; i<totalIOTensors; i++){
        std::string name = engine->getIOTensorName(i);

        if(engine->getTensorIOMode(name.c_str()) == mode){
            tensorNames.push_back(std::move(name));
        }
    }
    
    return tensorNames;
}


void logFullModelInfo(
    TrtLoggerAdaptor& logger,
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine
) {

    for (const std::string& name : getTensorNames(engine, nvinfer1::TensorIOMode::kINPUT)) {
        logger.logTensorDims(TrtSeverity::kINFO, name, engine->getTensorShape(name.c_str()));
    }

    for (const std::string& name : getTensorNames(engine, nvinfer1::TensorIOMode::kOUTPUT)) {
        logger.logTensorDims(TrtSeverity::kINFO, name, engine->getTensorShape(name.c_str()));
    }

    for (const std::string& name : getTensorNames(engine, nvinfer1::TensorIOMode::kNONE)) {
        logger.logTensorDims(TrtSeverity::kINFO, name, engine->getTensorShape(name.c_str()));
    }

    logger.log(TrtSeverity::kINFO, "Logging Layers Info of first and last few layers.\n");

    std::unique_ptr<nvinfer1::IEngineInspector> engineInspector(engine->createEngineInspector());
    int numLayers = engine->getNbLayers();
    
    for(int layerIdx=0; layerIdx<5; layerIdx++){
        logger.logConcatMessage(
            TrtSeverity::kINFO,
            "Layer Index:",
            layerIdx,
            '\t',
            engineInspector->getLayerInformation(layerIdx, nvinfer1::LayerInformationFormat::kONELINE),
            '\n'
        );
        
    }

}


Shape TrtDims2Shape(const nvinfer1::Dims& dims) {

    Shape shape;
    shape.dims.resize(dims.nbDims);

    for (int32_t i = 0; i < dims.nbDims; i++) {
        shape.dims[i] = static_cast<size_t>(dims.d[i]);
    }

    return shape;
}

nvinfer1::Dims ShapetoTrtDims(const Shape& shape) {

    nvinfer1::Dims dims;
    dims.nbDims = shape.rank();

    for (size_t i = 0; i < shape.rank(); i++) {
        dims.d[i] = static_cast<int64_t>(shape[i]);
    }

    return dims;
}
