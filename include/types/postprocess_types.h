#pragma once

#include <unordered_map>
#include <memory>
#include <opencv2/core/types.hpp>
#include "types/eigentensor_types.h"

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

    YoloBoxes _boxes;
    int _classId;
    float _conf;
    EigenTensor<float, 2> _maskCoeffs;

};


struct YoloSegMaskOutput{

};