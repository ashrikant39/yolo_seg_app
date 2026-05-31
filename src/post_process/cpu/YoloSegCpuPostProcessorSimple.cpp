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
#include "AppSettings.hpp"
#include "core/tensor.hpp"


YoloSegCpuPostProcessorSimple::YoloSegCpuPostProcessorSimple(const PostProcessorConfig& config):
    m_confidenceThresh(config.confThreshold),
    m_iouThresh(config.iouThreshold),
    m_maskThresh(config.maskThreshold),
    m_maxDetections(config.maxDetections) {

}

void YoloSegCpuPostProcessorSimple::process(
    const TensorViewMap& engineOutputBatch,
    std::vector<PostProcessOutput>& processedBatch,
    Logger& logger,
    cudaStream_t stream
){

    const std::string boxKey = SimplifiedYoloSettings::BOX_KEY;
    const std::string maskKey = SimplifiedYoloSettings::MASK_KEY;
    const std::string labelKey = SimplifiedYoloSettings::CLASS_LABEL;
    const std::string scoreKey = SimplifiedYoloSettings::OBJECTNESS;

    if (
        !engineOutputBatch.count(boxKey)     ||
        !engineOutputBatch.count(maskKey)    ||
        !engineOutputBatch.count(labelKey)   ||
        !engineOutputBatch.count(scoreKey)
    ) {
        logger.logConcatMessage(
            Severity::kERROR,
            "Missing output tensors");
        return;
    }

    const nvinfer1::Dims boxDims = engineOutputBatch.at(boxKey).dims; // .at()[B, NObjects, 4]
    const nvinfer1::Dims maskDims = engineOutputBatch.at(maskKey).dims; // [B, NObjects, H, W]

    const size_t batchSize = static_cast<size_t>(boxDims.d[0]);
    const size_t nBoxes = static_cast<size_t>(boxDims.d[1]);
    const size_t nCoeffs = static_cast<size_t>(boxDims.d[2]);

    const size_t maskH = static_cast<size_t>(maskDims.d[2]);
    const size_t maskW = static_cast<size_t>(maskDims.d[3]);

    const float *boxData = engineOutputBatch.at(boxKey).ptr<float>();
    const float *maskData = engineOutputBatch.at(maskKey).ptr<float>();
    const float *scoreData = engineOutputBatch.at(scoreKey).ptr<float>();
    const float *labelData = engineOutputBatch.at(labelKey).ptr<float>();

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

        NVTX_RANGE("ExtractBoxesAndScores");
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
        NVTX_POP();

        if (candBoxes.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections above threshold for batch item ", b, "\n");
            continue;
        }

        // Since Ultralytics doesn't support NMS for end-to-end models.
        // Applying non-maximal suppression, indices has to be of type [int]
        std::vector<int> nmsIndices;
        NVTX_RANGE("OpenCV_NMS");
        cv::dnn::NMSBoxes(
            candBoxes,
            candScores,
            m_confidenceThresh,
            m_iouThresh,
            nmsIndices,
            1.f,
            m_maxDetections);
        NVTX_POP();
        
        int detId = 0;
        processedBatch[b].detections.reserve(nmsIndices.size());

        if (nmsIndices.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections passed the NMS for batch item ", b, "\n");
            continue;
        }
        logger.logConcatMessage(Severity::kINFO, "Number of Detections: ", nmsIndices.size(), '\n');
        
        NVTX_RANGE("GET_MASKS_AND_BOXES_PER_IMAGE");
        for (int k : nmsIndices) {

            NVTX_RANGE("PROCESS_ONE_DETECTION");
            if (detId >= m_maxDetections) {
                break;
            }

            const int objIdx = candObjIndexes[k];
            cv::Rect2d boundingBox = candBoxes[k];
            const size_t label = candLabels[k];
            const double objScore = candScores[k];

            float *currMaskData = const_cast<float*>(maskData + idx4(b, objIdx, 0, 0, nBoxes, maskH, maskW));
            cv::Mat instMask(maskH, maskW, CV_32F, currMaskData);
            cv::Mat detMask8 = getRoIMaskFromRaw(instMask, boundingBox, origImgW, origImgH, m_maskThresh);
            Detection det;
            
            det.metadata.detectionId = detId;
            det.metadata.imgPath = processedBatch[b].metadata.imagePath;

            NVTX_POP();

            if (!getDetections(detMask8, boundingBox, label, objScore, det)) {
                logger.logConcatMessage(Severity::kINFO, "Couldn't Create Detections for frame: ", processedBatch[b].metadata.frameId, '\n');
                continue;
            }
            
            processedBatch[b].detections.push_back(std::move(det));

            ++detId;
        }
        NVTX_POP();
    }
}
