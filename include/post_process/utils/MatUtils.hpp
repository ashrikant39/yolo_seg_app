#pragma once

#include <opencv2/core.hpp>

#include "post_process/utils/PostProcessUtils.hpp"

cv::Mat computeInstanceMask(
    const float* protoBatch,  // [nProtoFeats, H, W] row-major contiguous
    int nProtoFeats,
    int maskH,
    int maskW,
    const float* maskCoeffs);

cv::Mat computeAllInstanceMasks(
    const float* protoBatch,
    const float* boxDataBatch,
    const std::vector<size_t>& candObjIndexes,
    const std::vector<int>& nmsIndices,
    size_t nMaskCoeffs,
    size_t maskW,
    size_t maskH,
    size_t maskStart,
    size_t nBoxes,
    size_t nCoeffs);

inline bool validateBox(double x1, double x2, double y1, double y2, double imageW, double imageH) { 
    return (x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0 && x1 < imageW && y1 < imageH && x2 < imageW && y2 < imageH);
}

cv::Mat getRoIMaskFromRaw(
    const cv::Mat& lowResRawMask,
    const cv::Rect2d& boundingBox,
    size_t maskW,
    size_t maskH,
    float maskThresh
);


bool getDetections(
    const cv::Mat& mask,
    const cv::Rect2d& boundingBox,
    size_t classLabel,
    double objectNess,
    Detection& retDetection
);