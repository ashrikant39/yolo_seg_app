#pragma once

#include <opencv2/core.hpp>

#include "post_process/utils/PostProcessUtils.hpp"

/**
 * @brief Computes one low-resolution instance mask from YOLO prototype features.
 *
 * @param protoBatch Contiguous prototype tensor for one batch item in CHW order.
 * @param nProtoFeats Number of prototype feature channels.
 * @param maskH Prototype mask height.
 * @param maskW Prototype mask width.
 * @param maskCoeffs Per-detection mask coefficients.
 * @return Floating-point mask logits for one detection.
 */
cv::Mat computeInstanceMask(
    const float* protoBatch,  // [nProtoFeats, H, W] row-major contiguous
    int nProtoFeats,
    int maskH,
    int maskW,
    const float* maskCoeffs);

/**
 * @brief Computes all selected instance masks for a batch item.
 *
 * @param protoBatch Contiguous prototype tensor for one batch item.
 * @param boxDataBatch Contiguous box tensor for one batch item.
 * @param candObjIndexes Candidate object indexes before NMS.
 * @param nmsIndices Indexes retained by NMS.
 * @param nMaskCoeffs Number of mask coefficients per candidate.
 * @param maskW Prototype mask width.
 * @param maskH Prototype mask height.
 * @param maskStart Offset of mask coefficients inside the box tensor row.
 * @param nBoxes Number of boxes in the batch item.
 * @param nCoeffs Number of coefficients per box row.
 * @return Matrix containing selected low-resolution masks.
 */
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

/**
 * @brief Checks whether a bounding box lies fully inside image bounds.
 */
inline bool validateBox(double x1, double x2, double y1, double y2, double imageW, double imageH) { 
    return (x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0 && x1 < imageW && y1 < imageH && x2 < imageW && y2 < imageH);
}

/**
 * @brief Crops and resizes a low-resolution raw mask into the detection ROI.
 *
 * @param lowResRawMask Raw mask logits/probabilities at prototype resolution.
 * @param boundingBox Detection bounding box in image coordinates.
 * @param maskW Prototype mask width.
 * @param maskH Prototype mask height.
 * @param maskThresh Threshold used to produce the binary ROI mask.
 * @return Binary ROI mask aligned to the bounding box.
 */
cv::Mat getRoIMaskFromRaw(
    const cv::Mat& lowResRawMask,
    const cv::Rect2d& boundingBox,
    size_t maskW,
    size_t maskH,
    float maskThresh
);

/**
 * @brief Populates a detection object from mask, box, class, and score values.
 *
 * @param mask Binary or probabilistic mask for the detection ROI.
 * @param boundingBox Detection bounding box in image coordinates.
 * @param classLabel Predicted class id.
 * @param objectNess Detection confidence score.
 * @param retDetection Detection object to populate.
 */
void getDetections(
    const cv::Mat& mask,
    const cv::Rect2d& boundingBox,
    size_t classLabel,
    double objectNess,
    Detection& retDetection
);
