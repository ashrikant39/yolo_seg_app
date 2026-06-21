#pragma once

#include <unordered_map>
#include <opencv2/core.hpp>

#include "post_process/utils/MatUtils.hpp"
#include "post_process/utils/PostProcessUtils.hpp"

static std::unordered_map<int, cv::Scalar> COLORS = {
    {0, cv::Scalar(255, 0, 0)},    // class 0: blue
    {1, cv::Scalar(0, 255, 0)},    // class 1: green
    {2, cv::Scalar(0, 0, 255)},    // class 2: red
};


cv::Mat drawRawMasksOnImage(
    const cv::Mat& image,
    const cv::Mat& rawInstanceMask,
    const Detection& detection,
    float maskThresh
);

cv::Mat drawDetectedMasksOnImage(
    const cv::Mat& image,
    const Detection& detection
);

cv::Mat drawContoursOnImage(
    const cv::Mat& image,
    const Detection& detection,
    int lineThickness
); 

cv::Mat drawBoundingBoxOnImage(
    const cv::Mat& image,
    const Detection& detection,
    int lineThickness
);