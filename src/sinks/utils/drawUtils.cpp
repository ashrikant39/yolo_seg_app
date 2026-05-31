#include "sinks/utils/drawUtils.hpp"
#include "core/cuda.hpp"

cv::Mat contourToMask(
    const std::vector<cv::Point2d>& contour,
    int imageW,
    int imageH
) {
    cv::Mat mask(imageH,imageW,CV_8UC1,cv::Scalar(0));

    std::vector<std::vector<cv::Point>> contours = {std::move(castContourToInt(contour))};

    cv::fillPoly(
        mask,
        contours,
        cv::Scalar(255)
    );

    return mask;
}


cv::Mat drawRawMasksOnImage(
    const cv::Mat& image,
    const cv::Mat& rawInstanceMask,
    const Detection& detection,
    float maskThresh
) {
    NVTX_RANGE("DRAW_MASK_ON_IMAGE");
    CV_Assert(rawInstanceMask.type() == CV_32F);

    if (detection.isNormalized) {
        throw std::runtime_error("Cannot use normalized boxes for drawing results");
    }

    cv::Scalar color = COLORS.count(detection.classLabel) ? COLORS[detection.classLabel] : cv::Scalar(0, 0, 0);
    cv::Mat mask8 = getRoIMaskFromRaw(rawInstanceMask, detection.boundingBox, image.cols, image.rows, maskThresh);   
    cv::Mat blendedImage, colorMask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    colorMask.setTo(color, mask8);
    cv::addWeighted(image, 0.7, colorMask, 0.3, 0.0, blendedImage);
    image.copyTo(blendedImage, ~mask8);
    NVTX_POP();

    return blendedImage;
}


cv::Mat drawDetectedMasksOnImage(
    const cv::Mat& image,
    const Detection& detection
) {
    NVTX_RANGE("DRAW_MASK_ON_IMAGE");
    
    if (detection.isNormalized) {
        throw std::runtime_error("Cannot use normalized boxes for drawing results");
    }

    cv::Mat instanceMask = contourToMask(detection.objectContour, image.cols, image.rows);
    cv::Scalar color = COLORS.count(detection.classLabel) ? COLORS[detection.classLabel] : cv::Scalar(0, 0, 0);

    cv::Mat blendedImage, colorMask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    colorMask.setTo(color, instanceMask);
    cv::addWeighted(image, 0.7, colorMask, 0.3, 0.0, blendedImage);
    image.copyTo(blendedImage, ~instanceMask);
    NVTX_POP();

    return blendedImage;
}



cv::Mat drawContoursOnImage(
    const cv::Mat& image,
    const Detection& detection,
    int lineThickness
) {
    
    NVTX_RANGE("DRAW_CONTOURS_ON_IMAGE");

    if (detection.isNormalized) {
        throw std::runtime_error("Cannot use normalized boxes for drawing results");
    }
    
    cv::Mat output = image.clone();
    std::vector<std::vector<cv::Point>> intContour = {std::move(castContourToInt(detection.objectContour))};

    cv::Scalar color = COLORS.count(detection.classLabel) ? COLORS[detection.classLabel] : cv::Scalar(0, 0, 0);
    cv::drawContours(output, intContour, -1, color, lineThickness);

    NVTX_POP();

    return output;
}


cv::Mat drawBoundingBoxOnImage(
    const cv::Mat& image,
    const Detection& detection,
    int lineThickness
) {
    NVTX_RANGE("DRAW_BOXES_ON_IMAGE");

    if (detection.isNormalized) {
        throw std::runtime_error("Cannot use normalized boxes for drawing results");
    }

    cv::Mat output = image.clone();
    cv::Rect boundingBox = castBoundingBoxToInt(detection.boundingBox);
    
    cv::Scalar color = COLORS.count(detection.classLabel) ? COLORS[detection.classLabel] : cv::Scalar(0, 0, 0);
    cv::rectangle(output, boundingBox, color, lineThickness);
    NVTX_POP();
    
    return output;
}
