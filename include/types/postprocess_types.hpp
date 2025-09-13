#pragma once

#include <unordered_map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include "types/eigentensor_types.hpp"

struct YoloBoxes{
    
    /*
    (x0, y0)------------------------------------
    |                                           |
    |                                           |
    |                                           |
    |-------------------------------------(x1, y1)
    */

    float _x0, _y0, _x1, _y1;

    YoloBoxes() = default;

    YoloBoxes(float x0, float y0, float x1, float y1):
    _x0(x0),
    _y0(y0),
    _x1(x1),
    _y1(y1){}

    cv::Rect2f convertBoxToOpenCVFormat(){
        return {_x0, _y0, _x1 - _x0, _y1 - _y0};
    }
};



struct YoloSegBoxOutput{

    YoloBoxes boxes;
    int classId;
    float conf;
    EigenTensor<float, 2> maskCoeffs;
};


struct YoloSegMaskOutput{
    cv::Mat masks;
};