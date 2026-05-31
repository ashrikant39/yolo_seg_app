#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "core/metadata.hpp"


struct DetectionMetaData {
    fs::path imgPath;
    uint64_t detectionId;
};

struct Detection {

    uint64_t classLabel;
    double objectness;
    cv::Rect2d boundingBox;
    std::vector<cv::Point2d> objectContour;
    DetectionMetaData metadata;
    bool isNormalized = false;

    bool serializeToByteArray(std::vector<uint8_t>& byteArray, uint64_t startPos = 0) const ;
    size_t getSerializedSize() const;
};

inline cv::Rect2d normalizeBox(const cv::Rect2d& box, size_t imageW, size_t imageH) {
    return {
        box.x / imageW,
        box.y / imageH,
        box.width / imageW,
        box.height / imageH
    };
}

inline cv::Rect2d normalizeBox(const cv::Rect& box, size_t imageW, size_t imageH) {
    return {
        static_cast<double>(box.x) / static_cast<double>(imageW),
        static_cast<double>(box.y) / static_cast<double>(imageH),
        static_cast<double>(box.width) / static_cast<double>(imageW),
        static_cast<double>(box.height) / static_cast<double>(imageH)
    };
}


inline cv::Rect2d denormalizeBox(const cv::Rect2d& box, size_t imageW, size_t imageH) {
    return {
        box.x * imageW,
        box.y * imageH,
        box.width * imageW,
        box.height * imageH
    };
}


inline cv::Rect denormalizeToIntBox(const cv::Rect2d& box, size_t imageW, size_t imageH) {
    return {
        static_cast<int>(box.x * imageW),
        static_cast<int>(box.y * imageH),
        static_cast<int>(box.width * imageW),
        static_cast<int>(box.height * imageH)
    };
}

inline void normalizeBoxInPlace(cv::Rect2d& box, size_t imageW, size_t imageH) {
    box.x /= imageW;
    box.y /= imageH;
    box.width /= imageW;
    box.height /= imageH;
}


inline void denormalizeBoxInPlace(cv::Rect2d& box, size_t imageW, size_t imageH) {
    box.x *= imageW;
    box.y *= imageH;
    box.width *= imageW;
    box.height *= imageH;
}

std::vector<cv::Point2d> normalizeContour(
    const std::vector<cv::Point2d>& contour,
    size_t imageW,
    size_t imageH
);

std::vector<cv::Point2d> normalizeContour(
    const std::vector<cv::Point>& contour,
    size_t imageW,
    size_t imageH
);


std::vector<cv::Point2d> denormalizeContour(
    const std::vector<cv::Point2d>& normalizedContour,
    size_t imageW,
    size_t imageH
);


std::vector<cv::Point> denormalizeToIntContour(
    const std::vector<cv::Point2d>& normalizedContour,
    size_t imageW,
    size_t imageH
);


void normalizeContourInPlace(
    std::vector<cv::Point2d>& contour,
    size_t imageW,
    size_t imageH
);


void denormalizeContourInPlace(
    std::vector<cv::Point2d>& contour,
    size_t imageW,
    size_t imageH
);

std::vector<cv::Point> castContourToInt(const std::vector<cv::Point2d>& contour);


cv::Rect castBoundingBoxToInt(const cv::Rect2d& boundingBox);

bool normalizeDetection(
    const Detection& inputDetection,
    Detection& resultDetection,
    size_t imageW,
    size_t imageH
);

bool normalizeDetectionInPlace(
    Detection& detection,
    size_t imageW,
    size_t imageH
);


bool denormalizeDetection(
    const Detection& inputDetection,
    Detection& resultDetection,
    size_t imageW,
    size_t imageH
);

bool denormalizeDetectionInPlace(
    Detection& detection,
    size_t imageW,
    size_t imageH
);

std::vector<uint8_t> serializeDetectionsToByteArray(const std::vector<Detection>& detections);

Detection deserializeFromByteArray(const std::vector<uint8_t>& bytes);


struct PostProcessOutput {
    std::vector<Detection> detections;
    FrameMetadata metadata;
};

