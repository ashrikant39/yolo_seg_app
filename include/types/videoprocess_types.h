#pragma once

#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct ImageBatchData{
    cv::Mat images;
    std::vector<fs::path> filePaths;
    
    ImageBatchData(int batchSize, int height, int width, int channels):
    images(cv::Mat({batchSize, channels, height, width}, CV_16F, cv::Scalar(0))){
        filePaths.reserve(batchSize);
    }
};
