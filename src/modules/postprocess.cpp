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


cv::Mat computeInstanceMask(
    const float* protoBatch,  // [nMaskCoeffs, H, W] row-major contiguous
    int nMaskCoeffs,
    int maskH,
    int maskW,
    const float* maskCoeffs) {

    const int hw = maskH * maskW;
    cv::Mat coeff(1, nMaskCoeffs, CV_32F, const_cast<float*>(maskCoeffs));
    cv::Mat protoFlat(nMaskCoeffs, hw, CV_32F, const_cast<float*>(protoBatch));
    cv::Mat logits;
    cv::gemm(coeff, protoFlat, 1.0, cv::Mat(), 0.0, logits);  // 1 x (H*W)

    cv::Mat mask(maskH, maskW, CV_32F);
    float* dst = mask.ptr<float>();
    const float* src = logits.ptr<float>();
    for (size_t i = 0; i < hw; ++i) {
        dst[i] = sigmoid(src[i]);
    }
    return mask;
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
    CudaTensorMap& modelOutputMap,
    const std::vector<fs::path>& batchFileNames,
    Logger& logger) {

    cudaDeviceSynchronize();

    // Lazy allocation of CPU buffers for output tensors.
    // We allocate based on the actual engine output tensor metadata provided at runtime.
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

    for (auto& [name, tensor] : modelOutputMap) {
        auto it = m_postProcessTensorMap.find(name);
        if (it == m_postProcessTensorMap.end()) {
            continue;
        }

        if (tensor.getDtype() == nvinfer1::DataType::kHALF){            
            castHalfToFloat(it->second.ptr<float>(), tensor.ptr<__half>(), tensor.getNumElements());
        }

        else if (tensor.getDtype() == nvinfer1::DataType::kFLOAT) {
            std::memcpy(it->second.ptr<float>(), tensor.ptr<float>(), tensor.getNumElements() * sizeof(float));
        }
        else {
            throw std::runtime_error("GPU Output : Wrong type");
        }
        
    }

    const std::string boxKey = ModelSettings::BOX_FEATURE_KEY;
    const std::string protoKey = ModelSettings::PROTO_MASK_KEY;

    if (!m_postProcessTensorMap.count(boxKey) || !m_postProcessTensorMap.count(protoKey)) {
        logger.logConcatMessage(
            Severity::kERROR,
            "Missing output tensors: need '",
            boxKey.c_str(),
            "' and '",
            protoKey.c_str(),
            "'\n");
        return;
    }

    const nvinfer1::Dims boxDims = m_postProcessTensorMap[boxKey].getDims();  // [B, NObjects, Box+Feature+Cls], Box+Feature+Cls = nCoeffs
    const nvinfer1::Dims protoDims = m_postProcessTensorMap[protoKey].getDims(); // [B, nMaskCoeffs, H, W] , nMaskCoeffs = Feature 

    const size_t batchSize = static_cast<size_t>(boxDims.d[0]);
    const size_t nBoxes = static_cast<size_t>(boxDims.d[1]);
    const size_t nCoeffs = static_cast<size_t>(boxDims.d[2]);
    const size_t nMaskCoeffs = static_cast<size_t>(protoDims.d[1]);
    const size_t maskH = static_cast<size_t>(protoDims.d[2]);
    const size_t maskW = static_cast<size_t>(protoDims.d[3]);

    const size_t nClsDims = PostProcessingOptions::NUM_CLS_DIMS;
    const size_t maskStart = YoloSegDecodeSettings::MASK_COEFF_START;
    const size_t nCoeffsExpected = nCoeffs - maskStart;

    std::unordered_map<int, cv::Scalar> classColors = {
        {0, cv::Scalar(255, 0, 0)},    // class 0: blue
        {1, cv::Scalar(0, 255, 0)},    // class 1: green
        {2, cv::Scalar(0, 0, 255)},    // class 2: red
    };

    if (nCoeffsExpected != nMaskCoeffs) {
        logger.logConcatMessage(
            Severity::kWARNING,
            "Mask coeff count (",
            nCoeffs - maskStart,
            ") != prototype channels (",
            nMaskCoeffs,
            "). Check MASK_COEFF_START / NUM_CLASSES vs engine.\n");
    }

    if (maskStart + nMaskCoeffs > nCoeffs) {
        logger.logConcatMessage(Severity::kERROR, "Invalid tensor layout: not enough channels for mask coeffs.\n");
        return;
    }

    fs::create_directories(m_resultsDir);

    const float* protoData = m_postProcessTensorMap[protoKey].ptr<float>();
    const float* boxData = m_postProcessTensorMap[boxKey].ptr<float>();

    for (size_t b = 0; b < batchSize; ++b) {

        if (b >= batchFileNames.size()) {
            break;
        }

        std::vector<cv::Rect2d> candBoxes;
        std::vector<float> candScores;
        std::vector<size_t> candOrigRow, candLabels;

        candBoxes.reserve(nBoxes);
        candScores.reserve(nBoxes);
        candOrigRow.reserve(nBoxes);
        candLabels.reserve(nBoxes);

        for (size_t i = 0; i < nBoxes; ++i) {
            // Decode box geometry from boxesXYWH.
            // Model outputs are assumed to be in pixel space of the preprocessed input.
            const float *boxStart = boxData + idx3(b, i, 0, nBoxes, nCoeffs);
            const float *scoreStart = boxData + idx3(b, i, 4, nBoxes, nCoeffs);
            const float *clsStart = boxData + idx3(b, i, 5, nBoxes, nCoeffs);

            const float objectness = *scoreStart;
            const size_t clsLabel = *clsStart;
            
            // OpenCV NMS requires boxes to be of double or int.
            const double x1 = boxStart[0];
            const double y1 = boxStart[1];
            const double x2 = boxStart[2];
            const double y2 = boxStart[3];

            if ( !validateBox(x1, x2, y1, y2, static_cast<double>(m_imageW), static_cast<double>(m_imageH)) ){
                continue;
            }

            const double bw = x2 - x1;
            const double bh = y2 - y1;
            
            if (objectness >= PostProcessingOptions::NMS_CONF_THRESH) {
                candBoxes.emplace_back(x1, y1, bw, bh);
                candOrigRow.push_back(i);
                candScores.push_back(objectness);
                candLabels.push_back(clsLabel);
            }
        }

        if (candBoxes.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections above threshold for batch item ", b, "\n");
            continue;
        }

        // Applying non-maximal suppression, indices has to be of type [int]
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(
            candBoxes,
            candScores,
            PostProcessingOptions::NMS_CONF_THRESH,
            PostProcessingOptions::NMS_IOU_THRESH,
            nmsIndices,
            1.f,
            PostProcessingOptions::NMS_MAX_DET);
        
        const fs::path& origImagePath = batchFileNames[b];
        const fs::path& outStem = batchFileNames[b].stem();
        const fs::path visPath = m_resultsDir / (outStem.string() + "_seg_vis.png");    
        cv::Mat canvas = cv::Mat::zeros(m_imageH, m_imageW, CV_8UC3);
        cv::Mat resizedImg;
        cv::resize(cv::imread(origImagePath, cv::IMREAD_COLOR), resizedImg, cv::Size(m_imageW, m_imageH));

        int detId = 0;

        for (int k : nmsIndices) {

            if (detId >= PostProcessingOptions::NMS_MAX_DET) {
                break;
            }

            const int origRow = candOrigRow[k];
            cv::Rect2d boundingBox = candBoxes[k];
            const size_t label = candLabels[k];
            cv::Scalar color = classColors.count(label) ? classColors[label] : cv::Scalar(0, 0, 0);

            std::vector<float> maskCoefficients(nMaskCoeffs);
            

            for (size_t j = 0; j < nMaskCoeffs; ++j) {
                size_t m_idx = idx3(b, origRow, maskStart + j, nBoxes, nCoeffs);
                maskCoefficients[j] = boxData[m_idx];
            }

            const size_t protoOff = idx4(b, 0, 0, 0, nMaskCoeffs, maskH, maskW);
            cv::Mat instMask = computeInstanceMask(protoData + protoOff, nMaskCoeffs, maskH, maskW, maskCoefficients.data());

            cv::Mat maskUp;
            cv::resize(instMask, maskUp, cv::Size(m_imageW, m_imageH), 0, 0, cv::INTER_LINEAR);

            cv::Mat binaryMask;
            cv::threshold(maskUp, binaryMask, PostProcessingOptions::MASK_THRESH, 1.0, cv::THRESH_BINARY);
            cv::Mat roiBinaryMask(binaryMask.size(), binaryMask.type(), cv::Scalar(0.0) );

            binaryMask(boundingBox).copyTo(roiBinaryMask(boundingBox));
            const fs::path maskPath = m_resultsDir / (outStem.string() + "_det" + std::to_string(detId) + "_mask.png");
            
            cv::Mat mask8, blendedImage, colorMask(resizedImg.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            roiBinaryMask.convertTo(mask8, CV_8U, 255.0);
            colorMask.setTo(color, mask8);
            cv::addWeighted(resizedImg, 0.7, colorMask, 0.3, 0.0, blendedImage);
            resizedImg.copyTo(blendedImage, ~mask8);
            cv::imwrite(maskPath, blendedImage);

            ++detId;
        }
    }
}
