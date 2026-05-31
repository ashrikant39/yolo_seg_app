#include "pre_process/modes/YoloSegCpuPreProcessor.hpp"

YoloSegCpuPreProcessor::YoloSegCpuPreProcessor(const PreProcessorConfig& config):
    m_scale(config.scalingFactor),
    m_mean(config.mean),
    m_isBGR(config.rgbOrdering == ChannelOrder::BGR),
    m_outImgH(config.resizeHeight),
    m_outImgW(config.resizeWidth) {

        if (config.ndimsInput != 4) {
            throw std::runtime_error("Incorrect ndimsInput for YoloSegCpuPreProcessor" + std::to_string(config.ndimsInput) + '\n');
        }

        m_nBatchDims = config.ndimsInput;
}

bool YoloSegCpuPreProcessor::process(
    const BatchFrameData& inputData,
    TensorViewMap& outputMap,
    const std::vector<std::string>& inputKeys
) {

    if (inputKeys.size() != 1 || inputKeys[0] != "images" ) {
        throw std::runtime_error("Invalid inputkeys parameter to YoloSegCpuPreProcessor");
    }

    int batchSize = static_cast<int>(inputData.images.size());
    int dims[] = {batchSize, NUM_IMG_CHANNELS, m_outImgH, m_outImgW};

    cv::float16_t *batchData = outputMap["images"].ptr<cv::float16_t>();

    if (!batchData) {
        throw std::runtime_error("Uninitialized Memory for input to the model.");
    }

    cv::Mat processedBatch(m_nBatchDims, dims, CV_16F, batchData);

    cv::dnn::blobFromImages(
        inputData.images,
        m_scale,
        cv::Size(m_outImgW, m_outImgH),
        cv::Scalar(m_mean),
        m_isBGR,
        false,
        CV_32F
    ).convertTo(processedBatch, CV_16F);

    return true;
}