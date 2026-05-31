#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>
#include <numeric>
#include <type_traits>
#include <opencv2/core.hpp>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "core/memory.hpp"


enum class DType {
    Float32,
    Float16,
    Int8,
    Int32,
    Bool,
    UInt8,
    BFloat16,
    Int64
};


inline size_t getSize(DType dtype){

    switch (dtype) {
        case DType::Float32: 
            return sizeof(float);

        case DType::Float16: 
            return sizeof(__half);

        case DType::Int8: 
            return sizeof(int8_t);

        case DType::Int32: 
            return sizeof(int32_t);

        case DType::Bool:
            return sizeof(bool);

        case DType::UInt8:
            return sizeof(uint8_t);

        case DType::BFloat16: 
            return sizeof(__nv_bfloat16);

        case DType::Int64: 
            return sizeof(int64_t);
            
        default: 
            throw std::runtime_error("Unknown data type");
    }
}


inline size_t getNumElements(const Shape& shape) {

    return std::accumulate(
        shape.dims.begin(), 
        shape.dims.end(), 
        static_cast<size_t>(1),
        std::multiplies<std::size_t>()
    )
}

enum class IOMode {
    Input,
    Output,
    None
};


struct Shape {

    std::vector<size_t> dims;

    size_t rank() const {
        return dims.size();
    }

    size_t operator[](size_t i) const {
        return dims[i];
    }
};


template <typename T>
struct TensorDtype;

template <>
struct TensorDtype<float> {
    static constexpr auto value = DType::Float32;
};

template <>
struct TensorDtype<__half> {
    static constexpr auto value = DType::Float16;
};

template <>
struct TensorDtype<cv::float16_t> {
    static constexpr auto value = DType::Float16;
};


template <>
struct TensorDtype<int8_t> {
    static constexpr auto value = DType::Int8;
};

template <>
struct TensorDtype<int32_t> {
    static constexpr auto value = DType::Int32;
};

template <>
struct TensorDtype<uint8_t> {
    static constexpr auto value = DType::UInt8;
};

template <>
struct TensorDtype<bool> {
    static constexpr auto value = DType::Bool;
};

template <typename, typename = void>
struct HasTensorDtype : std::false_type {};

template <typename T>
struct HasTensorDtype<T, std::void_t<decltype(TensorDtype<T>::value)>>
    : std::true_type {};


template <typename T>
inline void validateTensorDtype(DType actual) {

    static_assert(
        HasTensorDtype<T>::value,
        "Unsupported Tensor element type"
    );

    if (actual != TensorDtype<T>::value) {
        throw std::runtime_error("Requested pointer type does not match the tensor dtype");
    }
}



template <typename AllocPolicy>
class Tensor{

    public:

        using Storage = typename AllocPolicy::template Pointer<std::byte>;

        // Default constructor
        Tensor(): 
            m_dtype(DType::Float32),
            m_numElements(0),
            m_numBytes(0),
            m_shape{},
            m_mode(IOMode::None),
            m_ptr(nullptr) {}

        /**
         * @brief Construct a tensor wrapper around contiguous storage.
         * @param dtype Dtype metadata.
         * @param dims dimensions.
         * @param mode IO mode (input/output).
         */

        Tensor(DType dtype, const Shape& shape, IOMode mode):
            m_dtype(dtype),
            m_shape(shape),
            m_mode(mode),
            m_numElements(getNumElements(shape)),
            m_numBytes(getSize(m_dtype) * m_numElements),
            m_ptr(AllocPolicy::allocateBytes(m_numBytes)) 
            m_device(AllocPolicy::deviceType),
            m_memoryKind(AllocPolicy::memoryKind) {}

        /**
         * @brief Total number of elements count derived from shape.
         */
        size_t getNumElements() const {
            return m_numElements;
        }
        
        /**
         * @brief Total number of bytes occupied by the tensor.
         */
        size_t getTotalBytes() const {
            return m_numBytes;
        }
        
        /**
         * @brief Data type of the tensor.
         */
        DType getDtype() const {
            return m_dtype;
        }
        
        /**
         * @brief Shape type of the tensor.
         */
        Shape shape() const {
            return m_shape;
        }
        
        /**
         * @brief IOMode of the tensor.
         */
        IOMode getIOMode() const {
            return m_mode;
        }


        /**
         * @brief Mutable raw pointer to underlying contiguous storage.
         */
        void* rawPtr(){
            return m_ptr.get();
        }

        /**
         * @brief Const raw pointer to underlying contiguous storage.
         */
        const void* rawPtr() const {
            return m_ptr.get();
        }

        /**
         * @brief Cast the pointer to underlying contiguous storage to the original type.
         */
        template <typename T>
        T* ptr() {
            validateTensorDtype<T>(m_dtype);
            return reinterpret_cast<T*>(rawPtr());
        }

        /**
         * @brief Cast a const pointer to underlying contiguous storage to the original type.
         */
        template <typename T>
        const T* ptr() const {
            validateTensorDtype<T>(m_dtype);
            return reinterpret_cast<const T*>(rawPtr());
        }
        
        TensorView view() const {

            return TensorView {
                .data = m_ptr.get(),
                .type = m_dtype,
                .numElements = m_numElements,
                .totalBytes = m_numBytes,
                .shape = m_shape,
                .mode = m_mode,
                .device = m_device,
                .memoryKind = m_memoryKind
            };

        }
        
    private:
        DType m_dtype;
        size_t m_numElements, m_numBytes;
        Shape m_shape;
        IOMode m_mode;
        Storage m_ptr;
        
        // allocation parameters
        DeviceType m_device = DeviceType::UNSET;
        MemoryKind m_memoryKind = MemoryKind::UNSET;
};



struct TensorView {

    void *data = nullptr;
    DType type;
    size_t numElements = 0;
    size_t totalBytes = 0;
    Shape shape;
    IOMode mode;
    DeviceType device;
    MemoryKind memoryKind;

    /**
     * @brief Cast the pointer to underlying contiguous storage to the original type.
     */
    template <typename T>
    T* ptr() {
        validateTensorDtype<T>(type);
        return reinterpret_cast<T*>(data);
    }

    /**
     * @brief Cast a const pointer to underlying contiguous storage to the original type.
     */
    template <typename T>
    const T* ptr() const {
        validateTensorDtype<T>(type);
        return reinterpret_cast<const T*>(data);
    }
};


struct TensorInfo {
    Shape shape;
    DType dtype;
    IOMode mode;
};



using TensorMap = std::unordered_map<std::string, Tensor<MallocHostPolicy>>;
using CudaTensorMap = std::unordered_map<std::string, Tensor<CudaDevicePolicy>>;
using PinnedTensorMap = std::unordered_map<std::string, Tensor<PinnedHostPolicy>>;
using UnifiedMemoryTensorMap = std::unordered_map<std::string, Tensor<UnifiedMemoryPolicy>>;

using TensorViewMap = std::unordered_map<std::string, TensorView>;
using TensorInfoMap = std::unordered_map<std::string, TensorInfo>;

template <typename TensorMapType>
TensorViewMap getTensorViewMap(const TensorMapType& tensorMap) {

    TensorViewMap views;
    views.reserve(tensorMap.size());

    for (const auto& [name, tensor] : tensorMap) {
        views.emplace(name, tensor.view());
    }
    return views;
}

inline bool tensorMapInDevice(const TensorViewMap& tensorViewMap, DeviceType device) {

    for (const auto& [name, tensorView] : tensorViewMap) {
        if (tensorView.device != device) {
            return false;
        }
    }
    return true;
}


inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-std::max(-50.f, std::min(50.f, x))));
}

/** Linear index for [batch][i1][i2][i3] with dims [B, D1, D2, D3]. */
inline size_t idx4(int b, int i1, int i2, int i3, int D1, int D2, int D3){
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2 * D3) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2 * D3) +
           static_cast<size_t>(i2) * static_cast<size_t>(D3) +
           static_cast<size_t>(i3);
}

/** Linear index for [batch][i1][i2] with dims [B, D1, D2]. */
inline size_t idx3(int b, int i1, int i2, int D1, int D2){
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2) +
           static_cast<size_t>(i2);
}


/** Copy half-precision buffer to float32 (CPU-side buffers). */
inline void castHalfToFloat(float* dst, const __half* src, size_t numElements) {
    NVTX_RANGE("CASTING_DATA_TO_FLOAT");
    for (size_t i = 0; i < numElements; ++i) {
        dst[i] = __half2float(src[i]);
    }
    NVTX_POP();
}