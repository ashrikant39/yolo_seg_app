#pragma once

#include "types/tensor_types.hpp"
#include "types/eigentensor_types.hpp"
#include "types/postprocess_types.hpp"
#include "logger.hpp"
#include <vector>
#include "options.hpp"
#include <opencv2/core.hpp>

class PostProcessor{

    public:
        
        PostProcessor():
            _postProcessTensorMap(),
            _resultsDir(){}

        PostProcessor(
            const TensorMap<cv::float16_t>& inferenceTensorMap,
            const fs::path& resultsDir
        );

        ~PostProcessor();

        EigenTensorViewSharedPtr<float, 4> getTensorView4D(const std::string& tensorName);
        EigenTensorViewSharedPtrMap<float, 4> getTensorViewMap4D();
        void postProcessOutputs(const TensorMap<cv::float16_t> inferenceTensorMap, const std::vector<fs::path>& fileNames, Logger& logger);    

    private:
        //
        TensorMap<float> _postProcessTensorMap;
        fs::path _resultsDir;
};


// Copy float16 or bfloat16 data to float32
// originalDataPtr should contain the tensor to be copied
// copiedDataPtr should be an initialized pointer to a memory with totalElements
void copyDataToFloat32(
    cv::float16_t* sourceDataPtr,
    float* destDataPtr,
    size_t totalElements
);

// TensorViewSharedPtr<float, 4> getOutputTensorViewFromMemory4D(const std::string& outputName);
// TensorViewSharedPtrMap<float, 4> getOutputTensorViewMap();
// void postProcessOutputs(const fs::path& saveDir, const std::vector<fs::path>& imagePaths);
