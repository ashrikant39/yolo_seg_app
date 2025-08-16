#pragma once

enum class ComputeDataType : int{
    FLOAT32     =   0,
    FLOAT16     =   1,
    BFLOAT16    =   2,
    INT8        =   3,
    INT32       =   4,
    BOOL        =   5
};

enum class InferenceMode : int{
    ASYNC   =   0,
    SYNC    =   1
};
