#include "pre_process/modes/YoloSegCpuPreProcessor.hpp"
#include "pre_process/utils/PreProcessUtils.hpp"
#include "AppSettings.hpp"


YoloSegCpuPreProcessor::YoloSegCpuPreProcessor(const PreProcessorConfig& config):
    m_scale(config.imgScalingFactor),
    m_mean(config.imgMean),
    m_isBGR(config.imgRgbOrdering == ChannelOrderType::BGR),
    m_outImgH(config.imgResizeHeight),
    m_outImgW(config.imgResizeWidth),
    m_dtype(config.outputDataType) {

        if (std::find(
            YoloSegCpuPreProcessorSettings::supportedTypes.begin(),
            YoloSegCpuPreProcessorSettings::supportedTypes.end(),
            m_dtype) == YoloSegCpuPreProcessorSettings::supportedTypes.end()) {
            throw std::runtime_error("Data type not supported by YoloSegCpuPreProcessor");
        }

        const auto inputDims = config.ndimsOfInputs.find(std::string(YoloSegCpuPreProcessorSettings::ImageKey));
        if (inputDims == config.ndimsOfInputs.end() || inputDims->second != 4) {
            throw std::runtime_error("Incorrect input rank for YoloSegCpuPreProcessor");
        }

        m_nBatchDims = inputDims->second;
}

void YoloSegCpuPreProcessor::process(
    const BatchFrameData& inputData,
    TensorViewMap& resultBufferViews
) {

    int batchSize = static_cast<int>(inputData.images.size());
    auto imageTensor = resultBufferViews.at(
        std::string(YoloSegCpuPreProcessorSettings::ImageKey)
    );

    if (imageTensor.device != DeviceType::CPU) {
        throw std::runtime_error("Unsupported device for input preprocessing.");
    }

    cv::Mat processedBatch;

    if (m_dtype == DataType::Float16) {
        processedBatch = createBlob4D<cv::float16_t>(
            inputData.images,
            batchSize,
            StaticSettings::NUM_IMG_CHANNELS,
            m_outImgH,
            m_outImgW,
            m_mean,
            m_scale,
            m_isBGR,
            imageTensor
        );

    } else {
        processedBatch = createBlob4D<float>(
            inputData.images,
            batchSize,
            StaticSettings::NUM_IMG_CHANNELS,
            m_outImgH,
            m_outImgW,
            m_mean,
            m_scale,
            m_isBGR,
            imageTensor
        );

    }
}
