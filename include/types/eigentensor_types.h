#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

// Tensor View Types

template<typename TensorType, int rank>
using EigenTensorView = Eigen::TensorMap<Eigen::Tensor<TensorType, rank>>;

template<typename TensorType, int rank>
using EigenTensorViewSharedPtr = std::shared_ptr<EigenTensorView<TensorType, rank>>;

template<typename TensorType, int rank>
using EigenTensorViewSharedPtrMap = std::unordered_map<std::string, EigenTensorViewSharedPtr<TensorType, rank>>;

// Tensor Types
template <typename TensorType, int rank>
using EigenTensor = Eigen::Tensor<TensorType, rank>;

template<int rank>
using DSizeIndices = Eigen::DSizes<Eigen::Index, rank>;

template<int rank>
using ArrayIndices = Eigen::array<Eigen::Index, rank>;
