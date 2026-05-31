#include "post_process/utils/MatUtils.hpp"
#include "core/tensor.hpp"

cv::Mat computeInstanceMask(
    const float* protoBatch,  // [nMaskCoeffs, H, W] row-major contiguous
    int nMaskCoeffs,
    int maskW,
    int maskH,
    const float *maskCoeffs
) {
    
    NVTX_RANGE("INSTANCE_MASK_COMPUTE");
    const int hw = maskH * maskW;
    cv::Mat coeff(1, nMaskCoeffs, CV_32F, const_cast<float*>(maskCoeffs));
    cv::Mat protoFlat(nMaskCoeffs, hw, CV_32F, const_cast<float*>(protoBatch));
    cv::Mat logits;
    NVTX_RANGE("CV_GEMM");
    cv::gemm(coeff, protoFlat, 1.0, cv::Mat(), 0.0, logits);  // 1 x (H*W)
    NVTX_POP();
    cv::Mat mask(maskH, maskW, CV_32F);
    float* dst = mask.ptr<float>();
    const float* src = logits.ptr<float>();
    for (size_t i = 0; i < hw; ++i) {
        dst[i] = sigmoid(src[i]);
    }
    NVTX_POP();
    return mask;
}


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
    size_t nCoeffs
) { 

    NVTX_RANGE("computeAllInstanceMasks");
    size_t totalMasks = nmsIndices.size();
    cv::Mat coeffMat(totalMasks, nMaskCoeffs, CV_32F);
    size_t hw = maskH * maskW;

    for (size_t row = 0; row < totalMasks; ++row) {
        
        size_t k = nmsIndices[row];
        size_t objIdx = candObjIndexes[k];
        
        const float* coeffSrc = boxDataBatch + idx3(0, objIdx, maskStart, nBoxes, nCoeffs);

        std::memcpy(
            coeffMat.ptr<float>(row),
            coeffSrc,
            nMaskCoeffs * sizeof(float)
        );
    }

    cv::Mat protoFlat(nMaskCoeffs, hw, CV_32F, const_cast<float*>(protoBatch));
    cv::Mat maskLogits;

    cv::gemm(coeffMat, protoFlat, 1.0, cv::Mat(), 0.0, maskLogits); // (N, 32) @ (32, HW) -> (N, HW)

    CV_Assert(maskLogits.isContinuous());
    CV_Assert(maskLogits.type() == CV_32F);

    float* maskPtr = maskLogits.ptr<float>(0);

    for (size_t i = 0; i < totalMasks * hw ; i++) { 
        maskPtr[i] = sigmoid(maskPtr[i]);
    }

    NVTX_POP();
    return maskLogits;
}


cv::Mat getRoIMaskFromRaw(
    const cv::Mat& lowResRawMask,
    const cv::Rect2d& boundingBox,
    size_t maskW,
    size_t maskH,
    float maskThresh
) {

    NVTX_RANGE("ROI_MASK_COMPUTE");
    cv::Mat maskUp;
    cv::resize(lowResRawMask, maskUp, cv::Size(maskW, maskH), 0, 0, cv::INTER_LINEAR);
    cv::Mat binaryMask;
    cv::threshold(maskUp, binaryMask, maskThresh, 1.0, cv::THRESH_BINARY);
    cv::Mat roiBinaryMask(binaryMask.size(), binaryMask.type(), cv::Scalar(0.0));
    binaryMask(boundingBox).copyTo(roiBinaryMask(boundingBox));
    cv::Mat mask8;
    roiBinaryMask.convertTo(mask8, CV_8U, 255.0);
    NVTX_POP();
    return mask8;

}



bool getDetections(
    const cv::Mat& mask,
    const cv::Rect2d& boundingBox,
    size_t classLabel,
    double objectness,
    Detection& retDetection
) {
    NVTX_RANGE("GET_DETECTIONS");
    size_t imgH = mask.rows;
    size_t imgW = mask.cols;
    
    CV_Assert(mask.type() == CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        NVTX_POP();
        return false;
    }
    cv::Rect2d normedBox = normalizeBox(boundingBox, imgW, imgH);
    std::vector<cv::Point2d> normedContour = normalizeContour(contours[0], imgW, imgH);

    retDetection.classLabel = classLabel;
    retDetection.objectness = objectness;
    retDetection.boundingBox = normedBox;
    retDetection.objectContour = normedContour;
    NVTX_POP();
    return true;
}
