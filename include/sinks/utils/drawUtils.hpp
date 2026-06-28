#pragma once

#include <unordered_map>
#include <opencv2/core.hpp>

#include "post_process/utils/MatUtils.hpp"
#include "post_process/utils/PostProcessUtils.hpp"

/**
 * @brief Default class color lookup used by drawing sinks.
 */
static std::unordered_map<int, cv::Scalar> COLORS = {
    {0, cv::Scalar(255, 0, 0)},    // class 0: blue
    {1, cv::Scalar(0, 255, 0)},    // class 1: green
    {2, cv::Scalar(0, 0, 255)},    // class 2: red
};

/**
 * @brief Overlays a raw ROI mask for one detection on an image.
 */
cv::Mat drawRawMasksOnImage(
    const cv::Mat& image,
    const cv::Mat& rawInstanceMask,
    const Detection& detection,
    float maskThresh
);

/**
 * @brief Draws a detection contour/mask polygon on an image.
 */
cv::Mat drawDetectedMasksOnImage(
    const cv::Mat& image,
    const Detection& detection
);

/**
 * @brief Draws only the contour for one detection.
 */
cv::Mat drawContoursOnImage(
    const cv::Mat& image,
    const Detection& detection,
    int lineThickness
); 

/**
 * @brief Draws only the bounding box for one detection.
 */
cv::Mat drawBoundingBoxOnImage(
    const cv::Mat& image,
    const Detection& detection,
    int lineThickness
);
