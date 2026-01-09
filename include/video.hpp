#pragma once

#include <filesystem>
#include <vector>
#include <array>
#include "logger.hpp"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct ImageBatchData{

    cv::Mat images;
    std::vector<fs::path> filePaths;
    
    
    // CONSTRUCTOR
    ImageBatchData(int batchSize, int height, int width, int channels, cv::float16_t *ptr = nullptr){

        const int ndims = 4;
        const int dims[ndims] = {batchSize, channels, height, width};

        if(ptr){
            images = cv::Mat(ndims, dims, CV_16F, ptr);
        }
        else{
            images = cv::Mat(ndims, dims, CV_16F, cv::Scalar(0));
        }
        filePaths.reserve(batchSize);
    }
};



class ImageBatchLoader{

    public: 

        ImageBatchLoader();

        ImageBatchLoader(
            const fs::path& dirPath,
            size_t batchSize,
            size_t imgH,
            size_t imgW,
            Logger& logger,
            cv::float16_t *ptr = nullptr
        );

        size_t getTotalBatches() const {
            return (m_filesList.size() + m_batchSize - 1) / m_batchSize;
        }

        size_t getTotalImages() const {
            return m_filesList.size();
        }

        const std::vector<fs::path>& getFileNames() const {
            return m_filesList;
        }

        void loadBatchDataPreProcessed(
            int batchIdx, 
            Logger& logger, 
            double normFactorAddToScaled, 
            double normFactorScalingMul);

    private:
        std::vector<fs::path> m_filesList;
        size_t m_batchSize, m_imgH, m_imgW;
        ImageBatchData m_batchData;
};