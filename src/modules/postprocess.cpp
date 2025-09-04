#include "postprocess.h"
#include "settings.h"

void copyDataToFloat32(
    cv::float16_t* sourceDataPtr, 
    float* destDataPtr, 
    size_t totalElements){ 

        assert(reinterpret_cast<uintptr_t>(sourceDataPtr) % alignof(Eigen::half) == 0);

        EigenTensorView<Eigen::half, 1> sourceData(
            reinterpret_cast<Eigen::half*>(sourceDataPtr),
            totalElements
        );

        EigenTensorView<float, 1> copiedData(
            destDataPtr, 
            totalElements
        );

        for(int i=0; i<totalElements; i++){
            copiedData(i) = static_cast<float>(sourceData(i));
        }
}

// CONSTRUCTOR
PostProcessor::PostProcessor(
    const TensorMap<cv::float16_t>& inferenceTensorMap,
    const fs::path& resultsDir){

    for(const auto& [name, tensor]: inferenceTensorMap){
        
        _postProcessTensorMap[name] = {
            tensor.trtDtype,
            tensor.numElements,
            tensor.dims,
            tensor.mode,
            new float[tensor.numElements]
        };
    }
}


// DESTRUCTOR
PostProcessor::~PostProcessor(){
    
    for(const auto& [name, _tensor]: _postProcessTensorMap){
        delete[] _tensor.ptr;
    }
}


EigenTensorViewSharedPtr<float, 4> PostProcessor::getTensorView4D(const std::string& tensorName){
    
    if(!_postProcessTensorMap[tensorName].ptr){
        throw std::invalid_argument("Cannot create 4D tensor from null pointer");
    }

    DSizeIndices<4> arrayDims;

    for(int i=0; i<4; i++){
        arrayDims[i] = static_cast<Eigen::Index>(
            _postProcessTensorMap[tensorName].dims.d[i] >  0 ? _postProcessTensorMap[tensorName].dims.d[i] : 1);
    }

    return std::make_shared<EigenTensorView<float, 4>>(_postProcessTensorMap[tensorName].ptr, arrayDims);
}


EigenTensorViewSharedPtrMap<float, 4> PostProcessor::getTensorViewMap4D(){

    EigenTensorViewSharedPtrMap<float, 4> outputTensorViewMap;

    for(const auto& [name, _tensor]: _postProcessTensorMap){
        outputTensorViewMap.emplace(name, getTensorView4D(name));
    }

    return outputTensorViewMap;
}

// function to save results to files
void PostProcessor::postProcessOutputs(const TensorMap<cv::float16_t> inferenceTensorMap, const std::vector<fs::path>& fileNames, Logger& logger){
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

    for(const auto& [name, tensor]: inferenceTensorMap){
        
        copyDataToFloat32(
            tensor.ptr,
            _postProcessTensorMap[name].ptr,
            tensor.numElements
        );
    }
    
    EigenTensorViewSharedPtrMap<float, 4> outputTensors = getTensorViewMap4D();
    EigenTensorViewSharedPtr<float, 4> outputCoeffs = outputTensors[ModelSettings::BOX_FEATURE_KEY];

    auto dims0 = outputCoeffs->dimensions();    //  (16, 300, 38, 1) 
    int batchSize = dims0[0];
    int Nboxes = dims0[1];
    int NCoeffs = dims0[2];

    EigenTensorViewSharedPtr<float, 4> prototypeMasks = outputTensors[ModelSettings::PROTO_MASK_KEY];
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


    for(int batchIdx=0; batchIdx<batchSize; batchIdx++){

        EigenTensor<float, 2> boxes = outputCoeffs->slice(
            Eigen::DSizes<Eigen::Index, 4>{0, 0, 0, 0},
            Eigen::DSizes<Eigen::Index, 4>{batchSize, Nboxes, 4, 1}
        ).chip(0, 3).chip(batchIdx, 0).eval();    // (16, 300, 4, 1) -> (16, 300, 4) -> (300, 4)

        EigenTensor<float, 1> scores = outputCoeffs->slice(
            Eigen::DSizes<Eigen::Index, 4>{0, 0, 4, 0},
            Eigen::DSizes<Eigen::Index, 4>{batchSize, Nboxes, 1, 1}
        ).chip(0, 3).chip(0, 2).chip(batchIdx, 0).eval();    // (16, 300, 1, 1) -> (16, 300) -> (300)
        
        cv::Rect2f* boxData = reinterpret_cast<cv::Rect2f*>(boxes.data());

        std::vector<cv::Rect2f> cvBoxes(
            boxData,
            boxData + Nboxes
        );

        std::vector<float> scoresVec(
            scores.data(),
            scores.data() + Nboxes
        );
        
        scoresVec.size();
    }
}

// auto as a return type cannot make a function return multiple types through multiple paths
// Eigen::Tensor should get its rank as a compile time const.

// So, all CPU computations are to be done in float32
// Also, the rank of the tensor is to be kept 4.
// [[NOTE]]:  You have to use a dimension array of type Eigen::DSizes<Eigen::Index, 4> to use an array as the dims of an eigen tensor.
// TensorMap does not have a default constructor; so we can return a view tensor based on name instead of a view tensor map.