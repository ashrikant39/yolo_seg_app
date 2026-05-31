#pragma once

#include "pre_process/interface/PreProcessor.hpp"
#include "pre_process/config/PreProcessorConfig.hpp"

constexpr int NUM_IMG_CHANNELS = 3;

class YoloSegCpuPreProcessor : public PreProcessor {

    public:
        YoloSegCpuPreProcessor(const PreProcessorConfig& config);
        bool process(
            const BatchFrameData& inputData,
            TensorViewMap& outputMap,
            const std::vector<std::string>& inputKeys
        ) override;

    private:
        float m_scale, m_mean;
        bool m_isBGR;
        int m_nBatchDims;
        size_t m_outImgH, m_outImgW;
};  