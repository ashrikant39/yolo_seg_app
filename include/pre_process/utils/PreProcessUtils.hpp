#pragma once

#include <vector>
#include <type_traits>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


template <typename T>
cv::Mat createBlob4D(
    const std::vector<cv::Mat>& inputImages,
    int batchSize,
    int numChannels,
    int height,
    int width,
    float mean,
    float scale,
    bool isBGR,
    TensorView& resultTensor
) {

    static_assert(
        std::is_same_v<T, float> ||
        std::is_same_v<T, cv::float16_t>,
        "Unsupported blob type"
    );

    T *batchData = resultTensor.ptr<T>();

    if (!batchData) {
        throw std::runtime_error("Uninitialized Memory for input to the model.");
    }

    int dims[] = {batchSize, numChannels, height, width};
    cv::Mat processedBatch(4, dims, cv::DataType<T>::type, batchData);

    if constexpr (std::is_same_v<T, cv::float16_t>) {
        
        cv::Mat blobFP32;
        cv::dnn::blobFromImages(
            inputImages,
            blobFP32,
            scale,
            cv::Size(width, height),
            cv::Scalar(mean),
            isBGR,
            false,
            CV_32F
        );

        blobFP32.convertTo(processedBatch, cv::DataType<T>::type);

    } else {
        
        cv::dnn::blobFromImages(
            inputImages,
            processedBatch,
            scale,
            cv::Size(width, height),
            cv::Scalar(mean),
            isBGR,
            false,
            CV_32F
        );
    }

    return processedBatch;
}
