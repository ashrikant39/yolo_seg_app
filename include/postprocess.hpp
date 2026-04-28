#pragma once

#include "utils/tensor.hpp"
#include "logger.hpp"
#include <vector>
#include <filesystem>
#include "utils/options.hpp"
#include <opencv2/core.hpp>
#include "utils/cuda.hpp"
#include "utils/detection.hpp"

namespace fs = std::filesystem;

/**
 * @brief CPU-side post-processing for YOLO-seg style TensorRT outputs.
 *
 * Responsibilities:
 * - convert output tensors from FP16 to FP32 host buffers,
 * - decode boxes/scores/mask coefficients,
 * - run NMS,
 * - generate and save segmentation outputs.
 */
class PostProcessor{

    public:

        /**
         * @brief Construct post-processor with output directory and model input shape.
         * @param resultsDir Directory for saved masks/visualizations.
         * @param modelInputWidth Model input width in pixels.
         * @param modelInputHeight Model input height in pixels.
         */
        PostProcessor(
            const fs::path& resultsDir,
            int imageWidth,
            int imageHeight
        );

        void postProcessOutputs(
            HostTensorMap& modelOutputMap,
            const std::vector<fs::path>& batchFileNames,
            Logger& logger,
            bool saveDetsAsFile,
            bool drawMasksOnImage
        );

    private:
        TensorMap m_postProcessTensorMap;
        fs::path m_resultsDir;
        int m_imageW{0};
        int m_imageH{0};
};


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


void drawDetectedMasksOnImage(
    const cv::Mat& image,
    const fs::path& maskPath,
    const cv::Mat& instMask,
    const size_t resizeMaskH,
    const size_t resizeMaskW,
    const cv::Rect2d& boundingBox,
    const size_t label
);


bool getDetections(
    const cv::Mat& mask,
    const cv::Rect2d& boundingBox,
    size_t classLabel,
    double objectNess,
    Detection& retDetection
);