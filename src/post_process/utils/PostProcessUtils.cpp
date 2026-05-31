#include <numeric>

#include "post_process/utils/PostProcessUtils.hpp"
#include "post_process/utils/MatUtils.hpp"

std::vector<cv::Point2d> normalizeContour(const std::vector<cv::Point2d>& contour, size_t imageW, size_t imageH) {
    std::vector<cv::Point2d> normalizedContour;
    normalizedContour.reserve(contour.size());

    for (const cv::Point2d& pt : contour) {
        normalizedContour.emplace_back(pt.x / imageW, pt.y / imageH);
    }

    return normalizedContour;
}


std::vector<cv::Point2d> normalizeContour(const std::vector<cv::Point>& contour, size_t imageW, size_t imageH) {
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


std::vector<cv::Point2d> denormalizeContour(const std::vector<cv::Point2d>& normalizedContour, size_t imageW, size_t imageH) {
    std::vector<cv::Point2d> denormalizedContour;
    denormalizedContour.reserve(normalizedContour.size());

    for (const cv::Point2d& pt : normalizedContour) {
        denormalizedContour.emplace_back(pt.x * imageW, pt.y * imageH);
    }

    return denormalizedContour;
}


std::vector<cv::Point> denormalizeToIntContour(const std::vector<cv::Point2d>& normalizedContour, size_t imageW, size_t imageH) {
    std::vector<cv::Point> denormalizedContour;
    denormalizedContour.reserve(normalizedContour.size());

    for (const cv::Point2d& pt : normalizedContour) {
        denormalizedContour.emplace_back(
            cvRound(pt.x * imageW),
            cvRound(pt.y * imageH)
        );
    }

    return denormalizedContour;
}


void normalizeContourInPlace(std::vector<cv::Point2d>& contour, size_t imageW, size_t imageH) {

    for (cv::Point2d& pt : contour) {
        pt.x /= imageW;
        pt.y /= imageH;
    }
}

void denormalizeContourInPlace(std::vector<cv::Point2d>& contour, size_t imageW, size_t imageH) {
    for (cv::Point2d& pt : contour) {
        pt.x *= imageW;
        pt.y *= imageH;
    }
}

cv::Rect castBoundingBoxToInt(const cv::Rect2d& boundingBox) {
    return cv::Rect{
        cvRound(boundingBox.x),
        cvRound(boundingBox.y),
        cvRound(boundingBox.width),
        cvRound(boundingBox.height),
    };
}

std::vector<cv::Point> castContourToInt(const std::vector<cv::Point2d>& contour) {

    std::vector<cv::Point> intContour;
    intContour.reserve(contour.size());

    for (const auto& pt : contour) {
        intContour.emplace_back(
            cvRound(pt.x),
            cvRound(pt.y)
        );
    }

    return intContour;
}


bool normalizeDetection(const Detection& inputDetection, Detection& resultDetection, size_t imageW, size_t imageH) {
    
    if (inputDetection.isNormalized) {
        return false;
    }
    
    resultDetection.classLabel = inputDetection.classLabel;
    resultDetection.objectness = inputDetection.objectness;
    resultDetection.boundingBox = normalizeBox(inputDetection.boundingBox, imageW, imageH);
    resultDetection.objectContour = normalizeContour(inputDetection.objectContour, imageW, imageH);
    resultDetection.metadata = inputDetection.metadata;
    resultDetection.isNormalized = true;

    return true;
}


bool normalizeDetectionInPlace(Detection& detection, size_t imageW, size_t imageH) {

    if (detection.isNormalized) {
        return false;
    }

    normalizeBoxInPlace(detection.boundingBox, imageW, imageH);
    normalizeContourInPlace(detection.objectContour, imageW, imageH);
    detection.isNormalized = true;
    
    return true;
}


bool denormalizeDetection(const Detection& inputDetection, Detection& resultDetection, size_t imageW, size_t imageH) {
    
    if (!inputDetection.isNormalized) {
        return false;
    }
    
    resultDetection.classLabel = inputDetection.classLabel;
    resultDetection.objectness = inputDetection.objectness;
    resultDetection.boundingBox = denormalizeBox(inputDetection.boundingBox, imageW, imageH);
    resultDetection.objectContour = denormalizeContour(inputDetection.objectContour, imageW, imageH);
    resultDetection.metadata = inputDetection.metadata;
    resultDetection.isNormalized = false;

    return true;
}


bool denormalizeDetectionInPlace(Detection& detection, size_t imageW, size_t imageH) {

    if (!detection.isNormalized) {
        return false;
    }

    denormalizeBoxInPlace(detection.boundingBox, imageW, imageH);
    denormalizeContourInPlace(detection.objectContour, imageW, imageH);
    detection.isNormalized = false;
    
    return true;
}


size_t Detection::getSerializedSize() const {

    return  sizeof(uint64_t) +                                      // image filename size
            metadata.imgPath.string().size() + 1 +                  // image filename
            sizeof(uint64_t) +                                      // detection id
            sizeof(classLabel) +                                    // classLabel
            sizeof(objectness) +                                    // objectness
            sizeof(boundingBox.x) +                                 // box coordinates [x, y, w, h]
            sizeof(boundingBox.y) +
            sizeof(boundingBox.width) +
            sizeof(boundingBox.height) +
            sizeof(uint64_t) +                                      // number of countour points
            objectContour.size() * sizeof(double) * 2 +             // contour points [xi, yi]
            sizeof(bool);                                           // isNormalized
}

bool Detection::serializeToByteArray(std::vector<uint8_t>& byteArray, uint64_t startPos) const {

    uint64_t totalSize = getSerializedSize();

    if ( startPos > byteArray.size() || totalSize > byteArray.size() - startPos ) {
        return false;
    }

    uint8_t *ptr = byteArray.data() + startPos;

    auto writeValue = [&](const auto& value) {

        using T = std::decay_t<decltype(value)>;

        static_assert(
            std::is_trivially_copyable_v<T>,
            "writeValue() only supports trivially copyable types"
        );

        static_assert(
            !std::is_pointer_v<T>,
            "writeValue() does not support pointer types"
        );

        std::memcpy(ptr, &value, sizeof(T));
        ptr += sizeof(value);
    };


    auto writeArray = [&](const void *src, size_t size) {
        std::memcpy(ptr, src, size);
        ptr += size;
    };

    uint64_t totalContoursPts = objectContour.size();
    uint64_t fileNameSize = metadata.imgPath.string().size() + 1;

    writeValue(fileNameSize);
    writeArray(metadata.imgPath.c_str(), fileNameSize);

    writeValue(metadata.detectionId);
    writeValue(classLabel);
    writeValue(objectness);
    writeValue(boundingBox.x);
    writeValue(boundingBox.y);
    writeValue(boundingBox.width);
    writeValue(boundingBox.height);
    
    writeValue(totalContoursPts);
    writeArray(objectContour.data(), totalContoursPts * sizeof(objectContour[0]));
    writeValue(isNormalized);

    return true;
}



std::vector<uint8_t> serializeDetectionsToByteArray(const std::vector<Detection>& detections) {

    size_t totalSerializedSize = std::transform_reduce(
        detections.begin(),
        detections.end(),
        size_t{0},
        std::plus<>(),
        [](const Detection& d) noexcept {
            return d.getSerializedSize();   // or any function
        }
    );

    std::vector<uint8_t> serializedBytes(totalSerializedSize);
    size_t startPos = 0;

    for (const auto& det: detections) {

        if (!det.serializeToByteArray(serializedBytes, startPos)) {
            throw std::runtime_error("Serialization failed");
        }
        startPos += det.getSerializedSize();
        
    }

    return serializedBytes;
}

Detection deserializeFromByteArray(const std::vector<uint8_t>& bytes) {
        
    Detection det;
    const uint8_t *ptr = bytes.data();
    const uint8_t *end = bytes.data() + bytes.size();

    auto readValue = [&](auto& value) {
        using T = std::decay_t<decltype(value)>;

        static_assert(std::is_trivially_copyable_v<T>,
                      "readValue only supports trivially copyable types");
        static_assert(!std::is_pointer_v<T>,
                      "readValue does not support pointer types");

        if (static_cast<size_t>(end - ptr) < sizeof(T)) {
            throw std::runtime_error("deserializeFromByteArray: buffer too small");
        }

        std::memcpy(&value, ptr, sizeof(T));
        ptr += sizeof(T);

    };

    auto readArray = [&](void *dest, size_t size) {
        
        if (static_cast<size_t>(end - ptr) < size) {
            throw std::runtime_error("deserializeFromByteArray: buffer too small");
        }

        std::memcpy(dest, ptr, size);
        ptr += size;
    };

    uint64_t totalContourPoints = 0;
    uint64_t filenameSize = 0;

    readValue(filenameSize);
    std::vector<char> imgPathBuf(filenameSize);
    readArray(imgPathBuf.data(), filenameSize);
    
    if (imgPathBuf.back() != '\0') {
        throw std::runtime_error("Filename missing null terminator");
    }
    det.metadata.imgPath = std::filesystem::path(imgPathBuf.data());

    readValue(det.metadata.detectionId);
    readValue(det.classLabel);
    readValue(det.objectness);
    readValue(det.boundingBox.x);
    readValue(det.boundingBox.y);
    readValue(det.boundingBox.width);
    readValue(det.boundingBox.height);

    readValue(totalContourPoints);
    det.objectContour.resize(totalContourPoints);
    readArray(det.objectContour.data(), totalContourPoints * sizeof(cv::Point2d));
    readValue(det.isNormalized);

    if (ptr != end) {
        throw std::runtime_error("Excessive memory seen, might be corrupted.");
    }

    return det;
}