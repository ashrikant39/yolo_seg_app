#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>
#include <utils/memory.hpp>
#include <numeric>
#include <type_traits>
#include <opencv2/core.hpp>
#include <logger.hpp>


template <template <typename> class PtrType>
class Tensor{

    public:

        // Default constructor
        Tensor(): 
            m_trtDtype(nvinfer1::DataType::kFLOAT),
            m_numElements(0),
            m_dims{},
            m_mode(nvinfer1::TensorIOMode::kNONE),
            m_unqPtr(nullptr) {

                m_dims.nbDims = 0;
            }

        /**
         * @brief Construct a tensor wrapper around contiguous storage.
         * @param trtDtype TensorRT dtype metadata.
         * @param dims TensorRT dimensions.
         * @param mode TensorRT IO mode (input/output).
         */
        Tensor(nvinfer1::DataType trtDtype, const nvinfer1::Dims& dims, nvinfer1::TensorIOMode mode):
            m_trtDtype(trtDtype),
            m_dims(dims),
            m_mode(mode),
            m_numElements(std::accumulate(
                dims.d, 
                dims.d + dims.nbDims, 
                static_cast<size_t>(1),
                std::multiplies<>()
            )) {
                m_unqPtr = makeUniquePtr<std::byte, PtrType>(m_numElements * getElementSize(trtDtype));
            }

        /**
         * @brief Total flattened element count derived from `dims`.
         */
        size_t getNumElements() const {
            return m_numElements;
        }

        nvinfer1::DataType getDtype() const {
            return m_trtDtype;
        }

        nvinfer1::Dims getDims() const {
            return m_dims;
        }

        nvinfer1::TensorIOMode getIOMode() const {
            return m_mode;
        }


        /**
         * @brief Mutable raw pointer to underlying contiguous storage.
         */
        void* rawPtr(){
            return m_unqPtr.get();
        }

        /**
         * @brief Const raw pointer to underlying contiguous storage.
         */
        const void* rawPtr() const {
            return m_unqPtr.get();
        }

        template <typename T>
        T* ptr() {
            validateType<T>();
            return reinterpret_cast<T*>(m_unqPtr.get());
        }

        template <typename T>
        const T* ptr() const {
            validateType<T>();
            return reinterpret_cast<const T*>(m_unqPtr.get());
        }

    private:

        nvinfer1::DataType m_trtDtype;
        size_t m_numElements;
        nvinfer1::Dims m_dims;
        nvinfer1::TensorIOMode m_mode;
        PtrType<std::byte> m_unqPtr;

        template <typename T>
        void validateType() const {
            bool ok = false;

            if constexpr (std::is_same_v<T, float>) {
                ok = (m_trtDtype == nvinfer1::DataType::kFLOAT);

            } else if constexpr (std::is_same_v<T, __half>) {
                ok = (m_trtDtype == nvinfer1::DataType::kHALF);

            } else if constexpr (std::is_same_v<T, cv::float16_t>) {
                ok = (m_trtDtype == nvinfer1::DataType::kHALF);
                
            } else if constexpr (std::is_same_v<T, int8_t>) {
                ok = (m_trtDtype == nvinfer1::DataType::kINT8);

            } else if constexpr (std::is_same_v<T, int32_t>) {
                ok = (m_trtDtype == nvinfer1::DataType::kINT32);

            } else if constexpr (std::is_same_v<T, uint8_t>) {
                ok = (m_trtDtype == nvinfer1::DataType::kUINT8);

            } else if constexpr (std::is_same_v<T, bool>) {
                ok = (m_trtDtype == nvinfer1::DataType::kBOOL);

            }
            if (!ok) {
                throw std::runtime_error("Requested pointer type does not match TensorRT dtype");
            }
        }
};

using TensorMap = std::unordered_map<std::string, Tensor<UniquePtrToArray>>;
using DeviceTensorMap = std::unordered_map<std::string, Tensor<UniquePtrToDeviceArray>>;
using HostTensorMap = std::unordered_map<std::string, Tensor<UniquePtrToHostArray>>;

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