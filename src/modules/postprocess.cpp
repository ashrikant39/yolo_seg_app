#include "postprocess.hpp"
#include "settings.hpp"
#include "utils/tensor.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

cv::Mat computeInstanceMask(
    const float* protoBatch,  // [nMaskCoeffs, H, W] row-major contiguous
    int nMaskCoeffs,
    int maskW,
    int maskH,
    const float *maskCoeffs) {
    
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
    size_t nCoeffs) { 

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


cv::Mat getRoIMaskFromRaw(const cv::Mat& lowResRawMask, const cv::Rect2d& boundingBox, size_t maskW, size_t maskH) {

    NVTX_RANGE("ROI_MASK_COMPUTE");
    cv::Mat maskUp;
    cv::resize(lowResRawMask, maskUp, cv::Size(maskW, maskH), 0, 0, cv::INTER_LINEAR);
    cv::Mat binaryMask;
    cv::threshold(maskUp, binaryMask, PostProcessingOptions::MASK_THRESH, 1.0, cv::THRESH_BINARY);
    cv::Mat roiBinaryMask(binaryMask.size(), binaryMask.type(), cv::Scalar(0.0));
    binaryMask(boundingBox).copyTo(roiBinaryMask(boundingBox));
    cv::Mat mask8;
    roiBinaryMask.convertTo(mask8, CV_8U, 255.0);
    NVTX_POP();
    return mask8;

}


void drawDetectedMasksOnImage(
    const cv::Mat& image,
    const fs::path& maskPath,
    const cv::Mat& instMask,
    const size_t resizeMaskW,
    const size_t resizeMaskH,
    const cv::Rect2d& boundingBox,
    const size_t label
) {
    NVTX_RANGE("DRAW_DETECTIONS");
    CV_Assert(instMask.type() == CV_32F);
    CV_Assert(image.rows == resizeMaskH);
    CV_Assert(image.cols == resizeMaskW);

    std::unordered_map<int, cv::Scalar> classColors = {
        {0, cv::Scalar(255, 0, 0)},    // class 0: blue
        {1, cv::Scalar(0, 255, 0)},    // class 1: green
        {2, cv::Scalar(0, 0, 255)},    // class 2: red
    };

    cv::Scalar color = classColors.count(label) ? classColors[label] : cv::Scalar(0, 0, 0);
    cv::Mat mask8 = getRoIMaskFromRaw(instMask, boundingBox, resizeMaskW, resizeMaskH);   
    cv::Mat blendedImage, colorMask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    colorMask.setTo(color, mask8);
    cv::addWeighted(image, 0.7, colorMask, 0.3, 0.0, blendedImage);
    image.copyTo(blendedImage, ~mask8);
    cv::imwrite(maskPath, blendedImage);
    NVTX_POP();
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
    cv::Rect2d normedBox = normalizeBox(boundingBox, static_cast<double>(imgW), static_cast<double>(imgH));
    std::vector<cv::Point2d> normedContour = normalizeContour(contours[0], imgW, imgH);

    retDetection.classLabel = classLabel;
    retDetection.objectness = objectness;
    retDetection.normalizedBoundingBox = normedBox;
    retDetection.normalizedObjContour = normedContour;
    NVTX_POP();
    return true;
}


// CONSTRUCTOR — CPU float buffers for outputs only (decode / NMS / masks on host).
PostProcessor::PostProcessor(
    const fs::path& resultsDir,
    int imageWidth,
    int imageHeight)
    : m_resultsDir(resultsDir),
      m_imageW(imageWidth),
      m_imageH(imageHeight) {
}


void PostProcessor::postProcessOutputs(
    HostTensorMap& modelOutputMap,
    const std::vector<fs::path>& batchFileNames,
    Logger& logger,
    bool saveDetsAsFile,
    bool drawMasksOnImage) {

    // Lazy allocation of CPU buffers for output tensors.
    // We allocate based on the actual engine output tensor metadata provided at runtime.
    NVTX_RANGE("CreateInferTensors");
    if (m_postProcessTensorMap.empty()) {
        for (const auto& [name, tensor] : modelOutputMap) {
            if (tensor.getIOMode() != nvinfer1::TensorIOMode::kOUTPUT) {
                continue;
            }            
            m_postProcessTensorMap.emplace(
                name,
                Tensor<UniquePtrToArray>(
                    nvinfer1::DataType::kFLOAT,
                    tensor.getDims(),
                    nvinfer1::TensorIOMode::kOUTPUT)
            );
        }
    }
    NVTX_POP();

    NVTX_RANGE("CopyAndCastToCpu");
    for (auto& [m_name, m_tensor] : m_postProcessTensorMap) {
        
        const auto& outputTensor = modelOutputMap[m_name];

        if (outputTensor.getDtype() == nvinfer1::DataType::kHALF){            
            castHalfToFloat(m_tensor.ptr<float>(), outputTensor.ptr<__half>(), outputTensor.getNumElements());
        }

        else if (outputTensor.getDtype() == nvinfer1::DataType::kFLOAT) {
            std::memcpy(m_tensor.ptr<float>(), outputTensor.ptr<float>(), outputTensor.getNumElements() * sizeof(float));
        }
        else {
            throw std::runtime_error("GPU Output : Wrong type");
        }
        
    }
    NVTX_POP();

    // const std::string boxKey = ModelSettings::BOX_FEATURE_KEY;
    // const std::string protoKey = ModelSettings::PROTO_MASK_KEY;
    const std::string boxKey = SimpleModelSettings::BOX_KEY;
    const std::string maskKey = SimpleModelSettings::MASK_KEY;
    const std::string labelKey = SimpleModelSettings::CLASS_LABEL;
    const std::string scoreKey = SimpleModelSettings::OBJECTNESS;


    if (
        !m_postProcessTensorMap.count(boxKey)  ||
        !m_postProcessTensorMap.count(maskKey) ||
        !m_postProcessTensorMap.count(labelKey) ||
        !m_postProcessTensorMap.count(scoreKey)
    ) {
        logger.logConcatMessage(
            Severity::kERROR,
            "Missing output tensors");
        return;
    }

    // const nvinfer1::Dims boxDims = m_postProcessTensorMap[boxKey].getDims();  // [B, NObjects, Box+Feature+Cls], Box+Feature+Cls = nCoeffs
    // const nvinfer1::Dims protoDims = m_postProcessTensorMap[protoKey].getDims(); // [B, nMaskCoeffs, H, W] , nMaskCoeffs = Feature 
    const nvinfer1::Dims boxDims = m_postProcessTensorMap[boxKey].getDims(); // [B, NObjects, 4]
    const nvinfer1::Dims maskDims = m_postProcessTensorMap[maskKey].getDims(); // [B, NObjects, H, W]

    const size_t batchSize = static_cast<size_t>(boxDims.d[0]);
    const size_t nBoxes = static_cast<size_t>(boxDims.d[1]);
    const size_t nCoeffs = static_cast<size_t>(boxDims.d[2]);
    // const size_t nMaskCoeffs = static_cast<size_t>(protoDims.d[1]);
    const size_t maskH = static_cast<size_t>(maskDims.d[2]);
    const size_t maskW = static_cast<size_t>(maskDims.d[3]);

    const size_t nClsDims = PostProcessingOptions::NUM_CLS_DIMS;
    const size_t maskStart = YoloSegDecodeSettings::MASK_COEFF_START;
    const size_t nCoeffsExpected = nCoeffs - maskStart;

    // if (nCoeffsExpected != nMaskCoeffs) {
    //     logger.logConcatMessage(
    //         Severity::kWARNING,
    //         "Mask coeff count (",
    //         nCoeffs - maskStart,
    //         ") != prototype channels (",
    //         nMaskCoeffs,
    //         "). Check MASK_COEFF_START / NUM_CLASSES vs engine.\n");
    // }

    // if (maskStart + nMaskCoeffs > nCoeffs) {
    //     logger.logConcatMessage(Severity::kERROR, "Invalid tensor layout: not enough channels for mask coeffs.\n");
    //     return;
    // }

    fs::create_directories(m_resultsDir);

    const float *boxData = m_postProcessTensorMap[boxKey].ptr<float>();
    const float *maskData = m_postProcessTensorMap[maskKey].ptr<float>();
    const float *scoreData = m_postProcessTensorMap[scoreKey].ptr<float>();
    const float *labelData = m_postProcessTensorMap[labelKey].ptr<float>();

    for (size_t b = 0; b < batchSize; ++b) {

        if (b >= batchFileNames.size()) {
            break;
        }

        std::vector<cv::Rect2d> candBoxes;
        std::vector<float> candScores;
        std::vector<size_t> candObjIndexes, candLabels;

        candBoxes.reserve(nBoxes);
        candScores.reserve(nBoxes);
        candObjIndexes.reserve(nBoxes);
        candLabels.reserve(nBoxes);

        NVTX_RANGE("ExtractBoxesAndScores");
        for (size_t i = 0; i < nBoxes; ++i) {
            // Decode box geometry from boxesXYWH.
            // Model outputs are assumed to be in pixel space of the preprocessed input.
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

            if ( !validateBox(x1, x2, y1, y2, static_cast<double>(m_imageW), static_cast<double>(m_imageH)) ){
                continue;
            }

            const double bw = x2 - x1;
            const double bh = y2 - y1;
            
            if (objectness >= PostProcessingOptions::NMS_CONF_THRESH) {
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

        // Applying non-maximal suppression, indices has to be of type [int]
        std::vector<int> nmsIndices;
        NVTX_RANGE("OpenCV_NMS");
        cv::dnn::NMSBoxes(
            candBoxes,
            candScores,
            PostProcessingOptions::NMS_CONF_THRESH,
            PostProcessingOptions::NMS_IOU_THRESH,
            nmsIndices,
            1.f,
            PostProcessingOptions::NMS_MAX_DET);
        NVTX_POP();
        
        NVTX_RANGE("AfterNMS");
        const fs::path& origImagePath = batchFileNames[b];
        const fs::path& outStem = batchFileNames[b].stem();
        cv::Mat resizedImg;

        if (drawMasksOnImage) {
            logger.log(Severity::kINFO, "Drawing Masks on the Images.\n");
            cv::resize(cv::imread(origImagePath, cv::IMREAD_COLOR), resizedImg, cv::Size(m_imageW, m_imageH));
        }
        
        int detId = 0;
        
        std::vector<Detection> detections;
        detections.reserve(nmsIndices.size());

        if (nmsIndices.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections passed the NMS for batch item ", b, "\n");
            continue;
        }
        NVTX_POP();

        NVTX_RANGE("LogConcatMessage");
        logger.logConcatMessage(Severity::kINFO, "Number of Detections: ", nmsIndices.size(), '\n');
        NVTX_POP();
        
        // const size_t protoOffset = idx4(b, 0, 0, 0, nMaskCoeffs, maskH, maskW);
        // const size_t boxOffset = idx3(b, 0, 0, nBoxes, nCoeffs);

        // cv::Mat instanceMasks = computeAllInstanceMasks(
        //     protoData + protoOffset,
        //     boxData + boxOffset,
        //     candObjIndexes,
        //     nmsIndices,
        //     nMaskCoeffs,
        //     maskW,
        //     maskH,
        //     maskStart,
        //     nBoxes,
        //     nCoeffs);
        
        NVTX_RANGE("GET_MASKS_AND_BOXES_PER_IMAGE");

        for (int k : nmsIndices) {

            NVTX_RANGE("PROCESS_ONE_DETECTION");
            if (detId >= PostProcessingOptions::NMS_MAX_DET) {
                break;
            }

            const int objIdx = candObjIndexes[k];
            cv::Rect2d boundingBox = candBoxes[k];
            const size_t label = candLabels[k];
            const double objScore = candScores[k];

            // size_t maskCoeffOffSet = idx3(b, objIdx, maskStart, nBoxes, nCoeffs);
            // cv::Mat instMask = computeInstanceMask(protoData + protoOffset, nMaskCoeffs, maskW, maskH, boxData + maskCoeffOffSet);
            float *currMaskData = const_cast<float*>(maskData + idx4(b, objIdx, 0, 0, nBoxes, maskH, maskW));
            cv::Mat instMask(maskH, maskW, CV_32F, currMaskData);
            cv::Mat detMask8 = getRoIMaskFromRaw(instMask, boundingBox, m_imageW, m_imageH);
            Detection det;

            NVTX_POP();

            if (!getDetections(detMask8, boundingBox, label, objScore, det)) {
                continue;
            }

            detections.push_back(std::move(det));

            if (drawMasksOnImage) {
                const fs::path maskPath = m_resultsDir / (outStem.string() + "_det" + std::to_string(detId) + "_mask.png");
                drawDetectedMasksOnImage(resizedImg, maskPath, instMask, m_imageW, m_imageH, boundingBox, label);
            }
            ++detId;
        }
        NVTX_POP();

        if (saveDetsAsFile) {
                
            NVTX_RANGE("WRITE_DET");
            NVTX_RANGE("SERIALIZE_DETECTIONS");
            std::vector<uint8_t> bytes = serializeDetectionsToByteArray(detections);
            NVTX_POP();
            const fs::path detFilePath = m_resultsDir / (outStem.string() + "_detection" + ".bin");

            std::ofstream detFile(detFilePath, std::ios::out | std::ios::binary);

            if (!detFile) {
                throw std::runtime_error("Couldn't create the file");
            }
            detFile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            detFile.close();
            NVTX_POP();
        }
    }
}
