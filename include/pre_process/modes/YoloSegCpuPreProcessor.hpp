#pragma once

#include <array>

#include "pre_process/interface/PreProcessor.hpp"
#include "pre_process/config/PreProcessorConfig.hpp"
#include "memory_management/enums.hpp"


/**
 * @brief Static tensor-name and dtype settings for the CPU YOLO preprocessor.
 */
struct YoloSegCpuPreProcessorSettings {
    static constexpr std::string_view ImageKey = "images";
    static constexpr TensorGroup inputTensorGroup = TensorGroup::HostInput;
    static constexpr std::array<DataType, 2> supportedTypes = {DataType::Float16, DataType::Float32};
};

/**
 * @brief CPU OpenCV preprocessor for YOLO segmentation models.
 *
 * Produces the `images` tensor using OpenCV blob creation and writes directly
 * into the configured tensor buffer view.
 */
class YoloSegCpuPreProcessor : public PreProcessor {

    public:
        /**
         * @brief Construct from preprocessing configuration.
         * @param config Preprocessing shape, dtype, scaling, and channel settings.
         */
        YoloSegCpuPreProcessor(const PreProcessorConfig& config);

        /**
         * @copydoc PreProcessor::process
         */
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
