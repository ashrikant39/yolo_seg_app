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

namespace {

inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-std::max(-50.f, std::min(50.f, x))));
}

/** Linear index for [batch][i1][i2][i3] with dims [B, D1, D2, D3]. */
inline size_t idx4(int b, int i1, int i2, int i3, int D1, int D2, int D3){
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2 * D3) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2 * D3) +
           static_cast<size_t>(i2) * static_cast<size_t>(D3) +
           static_cast<size_t>(i3);
}

/** Linear index for [batch][i1][i2] with dims [B, D1, D2]. */
inline size_t idx3(int b, int i1, int i2, int D1, int D2){
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2) +
           static_cast<size_t>(i2);
}

cv::Mat computeInstanceMask(
    const float* protoBatch,  // [nm, H, W] row-major contiguous
    int nm,
    int maskH,
    int maskW,
    const float* maskCoeffs) {

    const int hw = maskH * maskW;
    cv::Mat coeff(1, nm, CV_32F, const_cast<float*>(maskCoeffs));
    cv::Mat protoFlat(nm, hw, CV_32F, const_cast<float*>(protoBatch));
    cv::Mat logits;
    cv::gemm(coeff, protoFlat, 1.0, cv::Mat(), 0.0, logits);  // 1 x (H*W)

    cv::Mat mask(maskH, maskW, CV_32F);
    float* dst = mask.ptr<float>();
    const float* src = logits.ptr<float>();
    for (int i = 0; i < hw; ++i) {
        dst[i] = sigmoid(src[i]);
    }
    return mask;
}

}  // namespace

// CONSTRUCTOR — CPU float buffers for outputs only (decode / NMS / masks on host).
PostProcessor::PostProcessor(
    const fs::path& resultsDir,
    int modelInputWidth,
    int modelInputHeight)
    : m_resultsDir(resultsDir),
      m_modelInputW(modelInputWidth),
      m_modelInputH(modelInputHeight) {
}


void PostProcessor::postProcessOutputs(
    CudaTensorMap<cv::float16_t>& inferenceTensorMap,
    const std::vector<fs::path>& batchFileNames,
    Logger& logger) {

    cudaDeviceSynchronize();

    // Lazy allocation of CPU buffers for output tensors.
    // We allocate based on the actual engine output tensor metadata provided at runtime.
    if (m_postProcessTensorMap.empty()) {
        for (const auto& [name, tensor] : inferenceTensorMap) {
            if (tensor.getIOMode() != nvinfer1::TensorIOMode::kOUTPUT) {
                continue;
            }            
            m_postProcessTensorMap.emplace(
                name,
                Tensor<float, UniquePtrToArray>(
                    nvinfer1::DataType::kFLOAT,
                    tensor.getDims(),
                    nvinfer1::TensorIOMode::kOUTPUT)
            );
        }
    }

    for (auto& [name, tensor] : inferenceTensorMap) {
        auto it = m_postProcessTensorMap.find(name);
        if (it == m_postProcessTensorMap.end()) {
            continue;
        }
        copyDataToFloat32(tensor.ptr(), it->second.ptr(), tensor.getNumElements());
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

    // Use Eigen views (slice/chip) for readability when decoding boxes/scores/mask coeffs.
    const nvinfer1::Dims boxDims = m_postProcessTensorMap[boxKey].getDims();  // [B, NObjects, Box+Feature+Cls], Box+Feature+Cls = nCoeffs
    const nvinfer1::Dims protoDims = m_postProcessTensorMap[protoKey].getDims(); // [B, nm, H, W] , nm = Feature 

    const int batchSize = static_cast<int>(boxDims.d[0]);
    const int nBoxes = static_cast<int>(boxDims.d[1]);
    const int nCoeffs = static_cast<int>(boxDims.d[2]);
    const int nm = static_cast<int>(protoDims.d[1]);
    const int maskH = static_cast<int>(protoDims.d[2]);
    const int maskW = static_cast<int>(protoDims.d[3]);

    const int nc = PostProcessingOptions::NUM_CLASSES;
    const int maskStart = YoloSegDecodeSettings::MASK_COEFF_START;
    const int nmExpected = nCoeffs - maskStart;

    if (nmExpected != nm) {
        logger.logConcatMessage(
            Severity::kWARNING,
            "Mask coeff count (",
            nCoeffs - maskStart,
            ") != prototype channels (",
            nm,
            "). Check MASK_COEFF_START / NUM_CLASSES vs engine.\n");
    }

    if (maskStart + nm > nCoeffs) {
        logger.logConcatMessage(Severity::kERROR, "Invalid tensor layout: not enough channels for mask coeffs.\n");
        return;
    }

    fs::create_directories(m_resultsDir);

    const float* protoData = m_postProcessTensorMap[protoKey].ptr();
    const float* boxData = m_postProcessTensorMap[boxKey].ptr();

    for (int b = 0; b < batchSize; ++b) {
        if (b >= static_cast<int>(batchFileNames.size())) {
            break;
        }

        // outputCoeffs: [B, Nboxes, NCoeffs, 1]
        // batchCoeffs2: [Nboxes, NCoeffs]
        // auto batchCoeffs3 = outputCoeffs->chip(b, 0);   // rank-3
        // auto batchCoeffs2 = batchCoeffs3.chip(2, 0);   // remove trailing dim=1 -> rank-2

        // First 4 channels are box params.
        // boxesXYWH: [Nboxes, 4]
        // auto boxesXYWH = batchCoeffs2.slice(
        //     Eigen::DSizes<Eigen::Index, 2>{0, 0},
        //     Eigen::DSizes<Eigen::Index, 2>{nBoxes, 4});

        // Score tensors: created lazily depending on layout.
        const bool hasObjAndCls = (maskStart == 4 + 1 + nc);
        const bool hasClsOnly = (maskStart == 4 + nc);

        // Eigen::Tensor<float, 2> objSliceT; // materialized via eval() when needed
        // Eigen::Tensor<float, 2> clsSliceT; // materialized via eval()

        int clsCount = 0;

        if (hasObjAndCls){
            clsCount = nc;
        }
        else if (hasClsOnly){
            clsCount = nc;
        }
        else{
            // Heuristic: objectness at 4, classes from 5 to maskStart.
            clsCount = std::min(nc, std::max(0, maskStart - 5));
        }

        std::vector<cv::Rect2d> candBoxes;
        std::vector<float> candScores;
        std::vector<int> candOrigRow;

        candBoxes.reserve(static_cast<size_t>(nBoxes));
        candScores.reserve(static_cast<size_t>(nBoxes));
        candOrigRow.reserve(static_cast<size_t>(nBoxes));

        for (int i = 0; i < nBoxes; ++i) {
            // Decode box geometry from boxesXYWH.
            // Model outputs are assumed to be in pixel space of the preprocessed input.
            const float *boxStart = boxData + idx3(b, i, 0, nBoxes, nCoeffs);
            const float *clsStart = boxData + idx3(b, i, 5, nBoxes, nCoeffs);

            size_t objIdx = idx3(b, i, 4, nBoxes, nCoeffs);
            const float objectness = boxData[objIdx];
            
            // OpenCV NMS requires boxes to be of double or int. 
            const double cx = boxStart[0];
            const double cy = boxStart[1];
            const double w = boxStart[2];
            const double h = boxStart[3];

            const double x1 = cx - 0.5 * w;
            const double y1 = cy - 0.5 * h;
            const double bw = std::max(0., w);
            const double bh = std::max(0., h);

            // Decode confidence and (optional) class via sigmoid logits.
            float best = 0.f;
            int bestId = 0;

            for (int c = 0; c < clsCount; ++c) {
                const float s = sigmoid(clsStart[c]);
                if (s > best) {
                    best = s;
                    bestId = c;
                }
            }

            float conf = best;
            if (hasObjAndCls || (!hasClsOnly && clsCount > 0)) {
                conf = sigmoid(objectness) * best;
            }

            if (conf >= PostProcessingOptions::NMS_CONF_THRESH) {
                candBoxes.emplace_back(x1, y1, bw, bh);
                candOrigRow.push_back(i);
                candScores.push_back(conf);
            }
        }

        if (candBoxes.empty()) {
            logger.logConcatMessage(Severity::kINFO, "No detections above threshold for batch item ", b, "\n");
            continue;
        }

        // Applying non-maximal suppression
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(
            candBoxes,
            candScores,
            PostProcessingOptions::NMS_CONF_THRESH,
            PostProcessingOptions::NMS_IOU_THRESH,
            nmsIndices,
            1.f,
            PostProcessingOptions::NMS_MAX_DET);

        const fs::path& outStem = batchFileNames[static_cast<size_t>(b)].stem();
        const fs::path visPath = m_resultsDir / (outStem.string() + "_seg_vis.png");

        cv::Mat canvas = cv::Mat::zeros(m_modelInputH, m_modelInputW, CV_8UC3);

        int detId = 0;
        for (int k : nmsIndices) {
            if (detId >= PostProcessingOptions::NMS_MAX_DET) {
                break;
            }

            const int origRow = candOrigRow[static_cast<size_t>(k)];

            std::vector<float> mcoef(static_cast<size_t>(nm));

            for (int j = 0; j < nm; ++j) {
                size_t m_idx = idx3(b, origRow, maskStart + j, nBoxes, nCoeffs);
                mcoef[static_cast<size_t>(j)] = boxData[m_idx];
            }

            const size_t protoOff = idx4(b, 0, 0, 0, nm, maskH, maskW);
            cv::Mat instMask = computeInstanceMask(protoData + protoOff, nm, maskH, maskW, mcoef.data());

            cv::Mat maskUp;
            cv::resize(instMask, maskUp, cv::Size(m_modelInputW, m_modelInputH), 0, 0, cv::INTER_LINEAR);

            cv::Mat bin;
            cv::threshold(maskUp, bin, 0.5, 1.0, cv::THRESH_BINARY);

            const uchar color[3] = {
                static_cast<uchar>(40 + (detId * 47) % 200),
                static_cast<uchar>(120 + (detId * 17) % 130),
                static_cast<uchar>(200 - (detId * 23) % 100)};

            for (int y = 0; y < canvas.rows; ++y) {
                cv::Vec3b* rowPtr = canvas.ptr<cv::Vec3b>(y);
                const float* m = bin.ptr<float>(y);
                for (int x = 0; x < canvas.cols; ++x) {
                    if (m[x] > 0.5f) {
                        rowPtr[x][0] = static_cast<uchar>((rowPtr[x][0] + color[0]) / 2);
                        rowPtr[x][1] = static_cast<uchar>((rowPtr[x][1] + color[1]) / 2);
                        rowPtr[x][2] = static_cast<uchar>((rowPtr[x][2] + color[2]) / 2);
                    }
                }
            }

            const fs::path maskPath = m_resultsDir / (outStem.string() + "_det" + std::to_string(detId) + "_mask.png");
            cv::Mat mask8;
            maskUp.convertTo(mask8, CV_8U, 255.0);
            cv::imwrite(maskPath.string(), mask8);

            ++detId;
        }

        cv::imwrite(visPath.string(), canvas);

        logger.logConcatMessage(
            Severity::kINFO,
            "Saved segmentation visual: ",
            visPath.string(),
            "\n");
    }
}
