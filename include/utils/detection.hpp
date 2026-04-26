#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

struct Detection {
    public:

        std::vector<uint8_t> serializeToByteArray() {
            size_t totalSize =  sizeof(size_t) +                                        // classLabel
                                sizeof(double) +                                        // objectness
                                sizeof(double) * 4 +                                    // box coordinates [x, y, w, h]
                                sizeof(size_t) +                                        // number of countour points
                                normalizedObjContour.size() * sizeof(double) * 2;       // contour points [xi, yi]
                                

            std::vector<uint8_t> bytes(totalSize);
            uint8_t *ptr = bytes.data();

            auto write = [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                std::memcpy(ptr, &value, sizeof(T));
                ptr += sizeof(T);
            };

            write(classLabel);
            write(objectness);
            write(normalizedBoundingBox.x);
            write(normalizedBoundingBox.y);
            write(normalizedBoundingBox.width);
            write(normalizedBoundingBox.height);
            write(normalizedObjContour.size());
            
            for (const cv::Point2d& pt : normalizedObjContour) {
                write(pt.x);
                write(pt.y);
            }
            return bytes;
        }

    public:
        size_t classLabel;
        double objectness;
        cv::Rect2d normalizedBoundingBox;
        std::vector<cv::Point2d> normalizedObjContour;
};


inline Detection deserializeFromByteArray(const std::vector<uint8_t>& bytes) {
        
    Detection det;
    const uint8_t *ptr = bytes.data();
    const uint8_t *end = bytes.data() + bytes.size();

    auto read = [&](auto& value) {
        using T = std::decay_t<decltype(value)>;

        if (ptr + sizeof(T) > end) {
            throw std::runtime_error("deserializeFromByteArray: buffer too small");
        }

        std::memcpy(&value, ptr, sizeof(T));
        ptr += sizeof(T);
    };

    size_t contourSize = 0;

    read(det.classLabel);
    read(det.objectness);
    read(det.normalizedBoundingBox.x);
    read(det.normalizedBoundingBox.y);
    read(det.normalizedBoundingBox.width);
    read(det.normalizedBoundingBox.height);
    read(contourSize);

    det.normalizedObjContour.reserve(contourSize);
    double x, y;

    for(size_t i = 0; i < contourSize; i++) {
        read(x);
        read(y);
        det.normalizedObjContour.emplace_back(x, y);
    }

    if (ptr != end) {
        throw std::runtime_error("deserializeFromByteArray: trailing bytes detected");
    }

    return det;
}


inline cv::Rect2d normalizeBox(const cv::Rect2d& box, double imageW, double imageH) {
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


inline cv::Rect2d denormalizeBox(const cv::Rect2d& box, double imageW, double imageH) {
    return {
        box.x * imageW,
        box.y * imageH,
        box.width * imageW,
        box.height * imageH
    };
}


inline cv::Rect denormalizeToIntBox(const cv::Rect2d& box, double imageW, double imageH) {
    return {
        static_cast<int>(box.x * imageW),
        static_cast<int>(box.y * imageH),
        static_cast<int>(box.width * imageW),
        static_cast<int>(box.height * imageH)
    };
}

inline void normalizeBoxInPlace(cv::Rect2d& box, double imageW, double imageH) {
    box.x /= imageW;
    box.y /= imageH;
    box.width /= imageW;
    box.height /= imageH;
}


inline void denormalizeBoxInPlace(cv::Rect2d& box, double imageW, double imageH) {
    box.x *= imageW;
    box.y *= imageH;
    box.width *= imageW;
    box.height *= imageH;
}


inline std::vector<cv::Point2d> normalizeContour(const std::vector<cv::Point2d>& contour, double imageW, double imageH) {
    std::vector<cv::Point2d> normalizedContour;
    normalizedContour.reserve(contour.size());

    for (const cv::Point2d& pt : contour) {
        normalizedContour.emplace_back(pt.x / imageW, pt.y / imageH);
    }

    return normalizedContour;
}


inline std::vector<cv::Point2d> normalizeContour(const std::vector<cv::Point>& contour, size_t imageW, size_t imageH) {
    std::vector<cv::Point2d> normalizedContour;
    normalizedContour.reserve(contour.size());

    for (const cv::Point2d& pt : contour) {
        normalizedContour.emplace_back(
            static_cast<double>(pt.x) / static_cast<double>(imageW),
            static_cast<double>(pt.y) / static_cast<double>(imageH)
        );
    }

    return normalizedContour;
}


inline std::vector<cv::Point2d> denormalizeContour(const std::vector<cv::Point2d>& normalizedContour, double imageW, double imageH) {
    std::vector<cv::Point2d> denormalizedContour;
    denormalizedContour.reserve(normalizedContour.size());

    for (const cv::Point2d& pt : normalizedContour) {
        denormalizedContour.emplace_back(pt.x * imageW, pt.y * imageH);
    }

    return denormalizedContour;
}


inline std::vector<cv::Point> denormalizeToIntContour(const std::vector<cv::Point2d>& normalizedContour, double imageW, double imageH) {
    std::vector<cv::Point> denormalizedContour;
    denormalizedContour.reserve(normalizedContour.size());

    for (const cv::Point2d& pt : normalizedContour) {
        denormalizedContour.emplace_back(
            static_cast<int>(pt.x * imageW),
            static_cast<int>(pt.y * imageH)
        );
    }

    return denormalizedContour;
}


inline void normalizeContourInPlace(std::vector<cv::Point2d>& contour, double imageW, double imageH) {

    for (cv::Point2d& pt : contour) {
        pt.x /= imageW;
        pt.y /= imageH;
    }
}

inline void denormalizeContourInPlace(std::vector<cv::Point2d>& contour, double imageW, double imageH) {
    for (cv::Point2d& pt : contour) {
        pt.x *= imageW;
        pt.y *= imageH;
    }
}