#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

// Tensor View Types

template<class TensorType, int rank>
using TensorView = Eigen::TensorMap<Eigen::Tensor<TensorType, rank>>;

template<class TensorType, int rank>
using TensorViewSharedPtr = std::shared_ptr<TensorView<TensorType, rank>>;

template<class TensorType, int rank>
using TensorViewSharedPtrMap = std::unordered_map<std::string, TensorViewSharedPtr<TensorType, rank>>;

// Tensor Types
template<int rank>
using FloatArray = Eigen::Tensor<float, rank>;


/*
Eigen::Index -> Index type
Eigen::DSizes<type of index, number of dims> -> Fixed size array of indices
*/

