#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>


#include "post_process/cpu/YoloSegCpuPostProcessorSimple.hpp"
#include "post_process/utils/MatUtils.hpp"
#include "core/tensor.hpp"


YoloSegCpuPostProcessorSimple::YoloSegCpuPostProcessorSimple(const PostProcessorConfig& config):
    m_confidenceThresh(config.confThreshold),
    m_iouThresh(config.iouThreshold),
    m_maskThresh(config.maskThreshold),
    m_maxDetections(config.maxDetections) {

}

void YoloSegCpuPostProcessorSimple::process(
    const TensorViewMap& engineOutputViews,
    std::vector<PostProcessOutput>& processedBatch,
    BaseLogger& logger,
    cudaStream_t stream
){

    const std::string boxKey(YoloSegCpuPostProcessorSimpleSettings::BoxKey);
    const std::string maskKey(YoloSegCpuPostProcessorSimpleSettings::MaskKey);
    const std::string labelKey(YoloSegCpuPostProcessorSimpleSettings::LabelKey);
    const std::string scoreKey(YoloSegCpuPostProcessorSimpleSettings::ScoreKey);

    if (
        !engineOutputViews.count(boxKey)     ||
        !engineOutputViews.count(maskKey)    ||
        !engineOutputViews.count(labelKey)   ||
        !engineOutputViews.count(scoreKey)
    ) {
        logger.logConcatMessage(
            Severity::kERROR,
            "Missing output buffer views",
            '\n'
        );
        return;
    }

    const Shape& boxDims = engineOutputViews.at(boxKey).shape;   // [B, NObjects, 4]
    const Shape& maskDims = engineOutputViews.at(maskKey).shape; // [B, NObjects, H, W]

    if (boxDims.rank() != 3 || maskDims.rank() != 4) {
        throw std::runtime_error("Unexpected modified YOLO segmentation output rank");
    }

    const size_t batchSize = boxDims[0];
    const size_t nBoxes = boxDims[1];
    const size_t maskH = maskDims[2];
    const size_t maskW = maskDims[3];

    if (processedBatch.size() < batchSize) {
        processedBatch.resize(batchSize);
    }

    const float *boxData = engineOutputViews.at(boxKey).ptr<float>();
    const float *maskData = engineOutputViews.at(maskKey).ptr<float>();
    const float *scoreData = engineOutputViews.at(scoreKey).ptr<float>();
    const float *labelData = engineOutputViews.at(labelKey).ptr<float>();

    for (size_t b = 0; b < batchSize; ++b) {

        std::vector<cv::Rect2d> candBoxes;
        std::vector<float> candScores;
        std::vector<size_t> candObjIndexes, candLabels;

        const size_t origImgW = processedBatch[b].metadata.originalWidth;
        const size_t origImgH = processedBatch[b].metadata.originalHeight;

        candBoxes.reserve(nBoxes);
        candScores.reserve(nBoxes);
        candObjIndexes.reserve(nBoxes);
        candLabels.reserve(nBoxes);

        
        for (size_t i = 0; i < nBoxes; ++i) {

            const float *currBoxData = boxData + idx3(b, i, 0, nBoxes, 4);
            const float *currScoreData = scoreData + idx3(b, i, 0, nBoxes, 1);
            const float *currLabelData = labelData + idx3(b, i, 0, nBoxes, 1);

            const float objectness = *currScoreData;
            const size_t clsLabel = *currLabelData;

            // OpenCV NMS requires boxes to be of double or int.
            const double x1 = currBoxData[0];
            const double y1 = currBoxData[1];
            const double x2 = currBoxData[2];
            const double y2 = currBoxData[3];

            if ( !validateBox(x1, x2, y1, y2, static_cast<double>(origImgW), static_cast<double>(origImgH)) ){
                continue;
            }

            const double bw = x2 - x1;
            const double bh = y2 - y1;

            if (objectness >= m_confidenceThresh) {
                candBoxes.emplace_back(x1, y1, bw, bh);
                candObjIndexes.push_back(i);
                candScores.push_back(objectness);
                candLabels.push_back(clsLabel);
            }

        }
        

        if (candBoxes.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections above threshold for batch item ", b, "\n");
            continue;
        }

        // Since Ultralytics doesn't support NMS for end-to-end models.
        // Applying non-maximal suppression, indices has to be of type [int]
        std::vector<int> nmsIndices;
        
        cv::dnn::NMSBoxes(
            candBoxes,
            candScores,
            m_confidenceThresh,
            m_iouThresh,
            nmsIndices,
            1.f,
            m_maxDetections);
        

        int detId = 0;
        processedBatch[b].detections.reserve(nmsIndices.size());

        if (nmsIndices.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections passed the NMS for batch item ", b, "\n");
            continue;
        }
        logger.logConcatMessage(Severity::kINFO, "Number of Detections: ", nmsIndices.size(), '\n');

        
        for (int k : nmsIndices) {

            
            if (detId >= m_maxDetections) {
                break;
            }

            const size_t objIdx = candObjIndexes[k];
            cv::Rect2d boundingBox = candBoxes[k];
            const size_t label = candLabels[k];
            const double objScore = candScores[k];

            float *currMaskData = const_cast<float*>(maskData + idx4(b, objIdx, 0, 0, nBoxes, maskH, maskW));
            cv::Mat instMask(maskH, maskW, CV_32F, currMaskData);
            cv::Mat detMask8 = getRoIMaskFromRaw(instMask, boundingBox, origImgW, origImgH, m_maskThresh);
            Detection det;

            det.metadata.detectionId = detId;
            det.metadata.imgPath = processedBatch[b].metadata.imagePath;
            getDetections(detMask8, boundingBox, label, objScore, det);

            if (det.objectContour.empty()) {
                logger.logConcatMessage(Severity::kINFO, "Couldn't get mask contour for frame: ", processedBatch[b].metadata.frameId, '\n');
            }

            processedBatch[b].detections.push_back(std::move(det));

            ++detId;
        }
        
    }
}
