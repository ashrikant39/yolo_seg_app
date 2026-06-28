#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "source/utils/frame.hpp"

namespace fs = std::filesystem;

/**
 * @brief Per-detection bookkeeping used by sinks and serializers.
 */
struct DetectionMetaData {
    /** Source image path associated with the detection. */
    fs::path imgPath;
    /** Stable detection id within the processed stream or batch. */
    uint64_t detectionId;
};

/**
 * @brief Normalized or image-space object detection with optional contour mask.
 */
struct Detection {

    /** Predicted class id. */
    uint64_t classLabel;
    /** Detection confidence score. */
    double objectness;
    /** Bounding box, either normalized or image-space depending on isNormalized. */
    cv::Rect2d boundingBox;
    /** Object contour points, either normalized or image-space depending on isNormalized. */
    std::vector<cv::Point2d> objectContour;
    /** Source image and detection id metadata. */
    DetectionMetaData metadata;
    /** True when box and contour coordinates are normalized to [0, 1]. */
    bool isNormalized = false;

    /**
     * @brief Serializes the detection into a caller-owned byte buffer.
     *
     * @param byteArray Destination byte buffer.
     * @param startPos Offset inside the destination buffer.
     * @return True when serialization succeeds.
     */
    bool serializeToByteArray(std::vector<uint8_t>& byteArray, uint64_t startPos = 0) const ;
    /**
     * @brief Returns the number of bytes required for binary serialization.
     */
    size_t getSerializedSize() const;
};

/**
 * @brief Converts an image-space floating-point box to normalized coordinates.
 */
inline cv::Rect2d normalizeBox(const cv::Rect2d& box, size_t imageW, size_t imageH) {
    return {
        box.x / imageW,
        box.y / imageH,
        box.width / imageW,
        box.height / imageH
    };
}

/**
 * @brief Converts an image-space integer box to normalized coordinates.
 */
inline cv::Rect2d normalizeBox(const cv::Rect& box, size_t imageW, size_t imageH) {
    return {
        static_cast<double>(box.x) / static_cast<double>(imageW),
        static_cast<double>(box.y) / static_cast<double>(imageH),
        static_cast<double>(box.width) / static_cast<double>(imageW),
        static_cast<double>(box.height) / static_cast<double>(imageH)
    };
}


/**
 * @brief Converts a normalized floating-point box to image-space coordinates.
 */
inline cv::Rect2d denormalizeBox(const cv::Rect2d& box, size_t imageW, size_t imageH) {
    return {
        box.x * imageW,
        box.y * imageH,
        box.width * imageW,
        box.height * imageH
    };
}


/**
 * @brief Converts a normalized floating-point box to integer image coordinates.
 */
inline cv::Rect denormalizeToIntBox(const cv::Rect2d& box, size_t imageW, size_t imageH) {
    return {
        static_cast<int>(box.x * imageW),
        static_cast<int>(box.y * imageH),
        static_cast<int>(box.width * imageW),
        static_cast<int>(box.height * imageH)
    };
}

/**
 * @brief Normalizes an image-space box in place.
 */
inline void normalizeBoxInPlace(cv::Rect2d& box, size_t imageW, size_t imageH) {
    box.x /= imageW;
    box.y /= imageH;
    box.width /= imageW;
    box.height /= imageH;
}


/**
 * @brief Denormalizes a box in place to image-space coordinates.
 */
inline void denormalizeBoxInPlace(cv::Rect2d& box, size_t imageW, size_t imageH) {
    box.x *= imageW;
    box.y *= imageH;
    box.width *= imageW;
    box.height *= imageH;
}

/**
 * @brief Converts image-space floating-point contour points to normalized coordinates.
 */
std::vector<cv::Point2d> normalizeContour(
    const std::vector<cv::Point2d>& contour,
    size_t imageW,
    size_t imageH
);

/**
 * @brief Converts image-space integer contour points to normalized coordinates.
 */
std::vector<cv::Point2d> normalizeContour(
    const std::vector<cv::Point>& contour,
    size_t imageW,
    size_t imageH
);


/**
 * @brief Converts normalized contour points to floating-point image coordinates.
 */
std::vector<cv::Point2d> denormalizeContour(
    const std::vector<cv::Point2d>& normalizedContour,
    size_t imageW,
    size_t imageH
);


/**
 * @brief Converts normalized contour points to integer image coordinates.
 */
std::vector<cv::Point> denormalizeToIntContour(
    const std::vector<cv::Point2d>& normalizedContour,
    size_t imageW,
    size_t imageH
);


/**
 * @brief Normalizes contour points in place.
 */
void normalizeContourInPlace(
    std::vector<cv::Point2d>& contour,
    size_t imageW,
    size_t imageH
);


/**
 * @brief Denormalizes contour points in place to image coordinates.
 */
void denormalizeContourInPlace(
    std::vector<cv::Point2d>& contour,
    size_t imageW,
    size_t imageH
);

/**
 * @brief Casts floating-point contour points to integer OpenCV points.
 */
std::vector<cv::Point> castContourToInt(const std::vector<cv::Point2d>& contour);


/**
 * @brief Casts a floating-point bounding box to an integer OpenCV rectangle.
 */
cv::Rect castBoundingBoxToInt(const cv::Rect2d& boundingBox);

/**
 * @brief Writes a normalized copy of a detection.
 */
bool normalizeDetection(
    const Detection& inputDetection,
    Detection& resultDetection,
    size_t imageW,
    size_t imageH
);

/**
 * @brief Normalizes a detection in place.
 */
bool normalizeDetectionInPlace(
    Detection& detection,
    size_t imageW,
    size_t imageH
);


/**
 * @brief Writes an image-space copy of a normalized detection.
 */
bool denormalizeDetection(
    const Detection& inputDetection,
    Detection& resultDetection,
    size_t imageW,
    size_t imageH
);

/**
 * @brief Denormalizes a detection in place.
 */
bool denormalizeDetectionInPlace(
    Detection& detection,
    size_t imageW,
    size_t imageH
);

/**
 * @brief Serializes a vector of detections into a compact byte array.
 */
std::vector<uint8_t> serializeDetectionsToByteArray(const std::vector<Detection>& detections);

/**
 * @brief Deserializes one detection from a byte array.
 */
Detection deserializeFromByteArray(const std::vector<uint8_t>& bytes);


/**
 * @brief Postprocessor result bundle for one source frame.
 */
struct PostProcessOutput {
    /** Detections generated for the frame. */
    std::vector<Detection> detections;
    /** Frame metadata propagated from the frame source. */
    FrameMetadata metadata;
};
