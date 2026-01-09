#pragma once

#include "utils/cudatensor.hpp"
#include "utils/eigentensor.hpp"
#include "utils/output.hpp"
#include "logger.hpp"
#include <vector>
#include "utils/options.hpp"
#include <opencv2/core.hpp>

class PostProcessor{

    public:
        
        PostProcessor():
            m_postProcessTensorMap(),
            m_resultsDir(){}

        PostProcessor(
            const TensorMap<cv::float16_t>& inferenceTensorMap,
            const fs::path& resultsDir
        );

        EigenTensorViewSharedPtr<float, 4> getTensorView4D(const std::string& tensorName);
        EigenTensorViewSharedPtrMap<float, 4> getTensorViewMap4D();
        void postProcessOutputs(CudaTensorMap<cv::float16_t>& inferenceTensorMap, const std::vector<fs::path>& fileNames, Logger& logger);    

    private:
        //
        TensorMap<float> m_postProcessTensorMap;
        fs::path m_resultsDir;
};

// TensorViewSharedPtr<float, 4> getOutputTensorViewFromMemory4D(const std::string& outputName);
// TensorViewSharedPtrMap<float, 4> getOutputTensorViewMap();
// void postProcessOutputs(const fs::path& saveDir, const std::vector<fs::path>& imagePaths);
