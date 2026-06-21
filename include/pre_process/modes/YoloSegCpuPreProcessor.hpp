#pragma once

#include <array>

#include "pre_process/interface/PreProcessor.hpp"
#include "pre_process/config/PreProcessorConfig.hpp"
#include "memory_management/enums.hpp"


struct YoloSegCpuPreProcessorSettings {
    static constexpr std::string_view ImageKey = "images";
    static constexpr TensorGroup inputTensorGroup = TensorGroup::HostInput;
    static constexpr std::array<DataType, 2> supportedTypes = {DataType::Float16, DataType::Float32};
};


class YoloSegCpuPreProcessor : public PreProcessor {

    public:
        YoloSegCpuPreProcessor(const PreProcessorConfig& config);
        void process(
            const BatchFrameData& inputData,
            TensorViewMap& resultBufferViews
        ) override;

    private:
        float m_scale, m_mean;
        bool m_isBGR;
        int m_nBatchDims;
        size_t m_outImgH, m_outImgW;
        DataType m_dtype;
};
