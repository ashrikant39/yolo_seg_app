#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <NvInfer.h>


struct BoundingBox {
    float x;        // Center x-coordinate
    float y;        // Center y-coordinate
    float width;    // Width of the box
    float height;   // Height of the box
    int classId;    // Class ID
    float boxConfidence; // Confidence score
};

struct YoloOutput {
    std::vector<BoundingBox> boundingBoxes; // All bounding boxes
    std::vector<cv::Mat> segmentationMaps;  // Segmentation maps
};