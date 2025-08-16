#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include "logger.h"

namespace fs = std::filesystem;

struct ImageBatchData{
    std::vector<cv::float16_t> dataBuffer;
    std::vector<fs::path> filePaths;
    
    ImageBatchData(size_t batchSize, size_t totalImgElements){
        dataBuffer.reserve(batchSize * totalImgElements);
        filePaths.reserve(batchSize);
    }
};


class VideoFromDirectory{

    public: 
        VideoFromDirectory(
            const fs::path& dirPath,
            size_t batchSize,
            size_t imgH,
            size_t imgW,
            Logger& logger
        );

        size_t getTotalBatches(){
            return (m_filesList.size() + m_batchSize - 1)/m_batchSize;
        }

        size_t getTotalImages(){
            return m_filesList.size();
        }

        const ImageBatchData& getBatchDataPreProcessed(
            int batchIdx, 
            Logger& logger, 
            double normFactorAddToScaled, 
            double normFactorScalingMul);

        ~VideoFromDirectory() = default;

    private:
        std::vector<fs::path> m_filesList;
        size_t m_batchSize, m_imgH, m_imgW;
        ImageBatchData m_batchData;
};


void preprocessImage(
    const cv::Mat& img, 
    cv::Mat& result, 
    size_t imgH, 
    size_t imgW, 
    double normFactorAddToScaled, 
    double normFactorScalingMul
);
