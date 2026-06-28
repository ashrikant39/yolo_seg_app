#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <opencv2/core.hpp>

#include "core/enums.hpp"
#include "core/memory.hpp"

/**
 * @brief Runtime tensor shape represented as size_t dimensions.
 */
struct Shape {
    std::vector<size_t> dims;

    /**
     * @brief Number of dimensions.
     */
    size_t rank() const {
        return dims.size();
    }

    /**
     * @brief Bounds-checked dimension access.
     */
    size_t operator[](size_t index) const {
        return dims.at(index);
    }
};

/**
 * @brief Return scalar byte size for a DataType.
 * @throws std::runtime_error for unsupported data types.
 */
inline size_t getSize(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
            return sizeof(float);
        case DataType::Float16:
            return sizeof(__half);
        case DataType::Int8:
            return sizeof(int8_t);
        case DataType::Int32:
            return sizeof(int32_t);
        case DataType::Bool:
            return sizeof(bool);
        case DataType::UInt8:
            return sizeof(uint8_t);
        case DataType::BFloat16:
            return sizeof(__nv_bfloat16);
        case DataType::Int64:
            return sizeof(int64_t);
        default:
            throw std::runtime_error("Unknown data type");
    }
}

/**
 * @brief Compute product of all shape dimensions.
 */
inline size_t getTotalNumElements(const Shape& shape) {
    return std::accumulate(
        shape.dims.begin(),
        shape.dims.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>()
    );
}

/**
 * @brief Compile-time mapping from C++ scalar type to DataType.
 */
template <typename T>
struct TensorDataType;

template <>
struct TensorDataType<float> {
    static constexpr auto value = DataType::Float32;
};

template <>
struct TensorDataType<__half> {
    static constexpr auto value = DataType::Float16;
};

template <>
struct TensorDataType<cv::float16_t> {
    static constexpr auto value = DataType::Float16;
};

template <>
struct TensorDataType<int8_t> {
    static constexpr auto value = DataType::Int8;
};

template <>
struct TensorDataType<int32_t> {
    static constexpr auto value = DataType::Int32;
};

template <>
struct TensorDataType<uint8_t> {
    static constexpr auto value = DataType::UInt8;
};

template <>
struct TensorDataType<bool> {
    static constexpr auto value = DataType::Bool;
};

/**
 * @brief Type trait indicating whether TensorDataType<T> is defined.
 */
template <typename, typename = void>
struct HasTensorDataType : std::false_type {};

template <typename T>
struct HasTensorDataType<T, std::void_t<decltype(TensorDataType<T>::value)>>
    : std::true_type {};

/**
 * @brief Validate that a requested pointer type matches a tensor DataType.
 * @throws std::runtime_error when actual does not match TensorDataType<T>.
 */
template <typename T>
inline void validateTensorDataType(DataType actual) {
    static_assert(
        HasTensorDataType<T>::value,
        "Unsupported tensor element type"
    );

    if (actual != TensorDataType<T>::value) {
        throw std::runtime_error("Requested pointer type does not match tensor dtype");
    }
}

/**
 * @brief Non-owning view over tensor storage.
 */
struct TensorView {
    void* data = nullptr;
    DataType type = DataType::Float32;
    size_t numElements = 0;
    size_t totalBytes = 0;
    Shape shape;
    IOMode mode = IOMode::None;
    DeviceType device = DeviceType::UNSET;
    MemoryType memoryType = MemoryType::UNSET;

    /**
     * @brief Return mutable typed pointer after dtype validation.
     */
    template <typename T>
    T* ptr() {
        validateTensorDataType<T>(type);
        return reinterpret_cast<T*>(data);
    }

    /**
     * @brief Return const typed pointer after dtype validation.
     */
    template <typename T>
    const T* ptr() const {
        validateTensorDataType<T>(type);
        return reinterpret_cast<const T*>(data);
    }
};

/**
 * @brief Owning tensor buffer parameterized by an allocation policy.
 *
 * AllocPolicy decides where bytes are allocated and which deleter is used.
 */
template <typename AllocPolicy>
class Tensor {
    public:
        using Storage = typename AllocPolicy::BytePointer;

        Tensor() = default;

        /**
         * @brief Allocate tensor storage.
         * @param dtype Scalar data type.
         * @param shape Tensor shape.
         * @param mode Input/output mode.
         */
        Tensor(DataType dtype, const Shape& shape, IOMode mode):
            m_dtype(dtype),
            m_numElements(getTotalNumElements(shape)),
            m_numBytes(getSize(dtype) * m_numElements),
            m_shape(shape),
            m_mode(mode),
            m_ptr(AllocPolicy::allocateBytes(m_numBytes)),
            m_device(AllocPolicy::deviceType),
            m_memoryType(AllocPolicy::memoryType) {}

        /**
         * @brief Number of tensor elements.
         */
        size_t numel() const {
            return m_numElements;
        }

        /**
         * @brief Total storage size in bytes.
         */
        size_t getTotalBytes() const {
            return m_numBytes;
        }

        /**
         * @brief Tensor scalar data type.
         */
        DataType getDataType() const {
            return m_dtype;
        }

        /**
         * @brief Tensor shape.
         */
        Shape shape() const {
            return m_shape;
        }

        /**
         * @brief Tensor IO mode.
         */
        IOMode getIOMode() const {
            return m_mode;
        }

        /**
         * @brief Mutable raw storage pointer.
         */
        void* rawPtr() {
            return m_ptr.get();
        }

        /**
         * @brief Const raw storage pointer.
         */
        const void* rawPtr() const {
            return m_ptr.get();
        }

        // template <typename T>
        // T* ptr() {
        //     validateTensorDataType<T>(m_dtype);
        //     return reinterpret_cast<T*>(rawPtr());
        // }

        // template <typename T>
        // const T* ptr() const {
        //     validateTensorDataType<T>(m_dtype);
        //     return reinterpret_cast<const T*>(rawPtr());
        // }

        /**
         * @brief Create a non-owning view over this tensor.
         */
        TensorView view() const {
            return TensorView{
                .data = m_ptr.get(),
                .type = m_dtype,
                .numElements = m_numElements,
                .totalBytes = m_numBytes,
                .shape = m_shape,
                .mode = m_mode,
                .device = m_device,
                .memoryType = m_memoryType
            };
        }

    private:
        DataType m_dtype = DataType::Float32;
        size_t m_numElements = 0;
        size_t m_numBytes = 0;
        Shape m_shape;
        IOMode m_mode = IOMode::None;
        Storage m_ptr;
        DeviceType m_device = DeviceType::UNSET;
        MemoryType m_memoryType = MemoryType::UNSET;
};

struct TensorSpec {
    Shape shape;
    DataType dtype = DataType::Float32;
    IOMode mode = IOMode::None;
};

using HostTensorMap = std::unordered_map<std::string, Tensor<MallocHostPolicy>>;
using CudaTensorMap = std::unordered_map<std::string, Tensor<CudaDevicePolicy>>;
using PinnedHostTensorMap = std::unordered_map<std::string, Tensor<PinnedHostPolicy>>;
using UnifiedTensorMap = std::unordered_map<std::string, Tensor<UnifiedMemoryPolicy>>;

using TensorStorageMap = std::variant<
    HostTensorMap,
    CudaTensorMap,
    PinnedHostTensorMap,
    UnifiedTensorMap
>;
using TensorViewMap = std::unordered_map<std::string, TensorView>;
using TensorSpecMap = std::unordered_map<std::string, TensorSpec>;

/**
 * @brief Build a TensorViewMap from an owning tensor map.
 */
template <typename TensorMapType>
TensorViewMap getTensorViews(const TensorMapType& tensorMap) {
    TensorViewMap views;
    views.reserve(tensorMap.size());

    for (const auto& [name, tensor] : tensorMap) {
        views.emplace(name, tensor.view());
    }
    return views;
}

/**
 * @brief Check if every tensor view is on a given device.
 */
inline bool TensorViewsOnDevice(const TensorViewMap& TensorViews, DeviceType device) {
    for (const auto& [name, TensorView] : TensorViews) {
        if (TensorView.device != device) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Check if every tensor view uses a given memory type.
 */
inline bool TensorViewsInMemory(const TensorViewMap& TensorViews, MemoryType memoryType) {
    for (const auto& [name, TensorView] : TensorViews) {
        if (TensorView.memoryType != memoryType) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Numerically clamped sigmoid helper.
 */
inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-std::clamp(x, -50.f, 50.f)));
}

/**
 * @brief Compute flattened row-major index for a 4D tensor.
 */
inline size_t idx4(int b, int i1, int i2, int i3, int D1, int D2, int D3) {
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2 * D3) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2 * D3) +
           static_cast<size_t>(i2) * static_cast<size_t>(D3) +
           static_cast<size_t>(i3);
}

/**
 * @brief Compute flattened row-major index for a 3D tensor.
 */
inline size_t idx3(int b, int i1, int i2, int D1, int D2) {
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2) +
           static_cast<size_t>(i2);
}

/**
 * @brief Convert CUDA half values to float.
 */
inline void castHalfToFloat(float* dst, const __half* src, size_t numElements) {
    for (size_t i = 0; i < numElements; ++i) {
        dst[i] = __half2float(src[i]);
    }
}
