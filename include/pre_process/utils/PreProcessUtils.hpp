#pragma once

#include <vector>
#include <type_traits>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


/**
 * @brief Creates a 4D NCHW OpenCV blob directly over a tensor buffer view.
 *
 * The destination tensor must already own enough memory for
 * `batchSize * numChannels * height * width` elements of type `T`.
 *
 * @tparam T Output element type. Supported types are `float` and `cv::float16_t`.
 * @param inputImages Input OpenCV images for the batch.
 * @param batchSize Number of images in the batch.
 * @param numChannels Number of output channels.
 * @param height Output tensor height.
 * @param width Output tensor width.
 * @param mean Mean value passed to OpenCV blob conversion.
 * @param scale Scale factor passed to OpenCV blob conversion.
 * @param isBGR Whether OpenCV should keep BGR channel order.
 * @param resultTensor Destination tensor view.
 * @return OpenCV matrix header pointing at the destination tensor storage.
 */
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
