#pragma once

#include <filesystem>
#include <vector>
#include <array>
#include "logger.hpp"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/**
 * @brief Container for one preprocessed input batch in NCHW FP16 layout.
 *
 * `images` is stored as a 4D OpenCV matrix with dimensions:
 * [batch, channels, height, width].
 */
struct ImageBatchData{

    cv::Mat images;
    std::vector<fs::path> filePaths;
    
    
    /**
     * @brief Construct a batch container with optional external backing buffer.
     * @param batchSize Number of images in a batch.
     * @param height Input height.
     * @param width Input width.
     * @param channels Input channels (typically 3).
     * @param ptr Optional external FP16 buffer. If null, storage is internally allocated.
     */
    ImageBatchData(int batchSize, int height, int width, int channels, cv::float16_t *ptr = nullptr){

        const int ndims = 4;
        const int dims[ndims] = {batchSize, channels, height, width};

        if(ptr){
            images = cv::Mat(ndims, dims, CV_16F, ptr);
        }
        else{
            images = cv::Mat(ndims, dims, CV_16F, cv::Scalar(0));
        }
        filePaths.reserve(batchSize);
    }
};

/**
 * @brief Loads images from a directory and prepares model input batches.
 *
 * Images are read via OpenCV and converted to NCHW FP16 blobs suitable for inference.
 */
class ImageBatchLoader{

    public: 

        /**
         * @brief Default constructor for delayed initialization.
         */
        ImageBatchLoader();

        /**
         * @brief Build a loader over an image directory.
         * @param dirPath Directory containing image files.
         * @param batchSize Number of images per model batch.
         * @param imgH Model input height.
         * @param imgW Model input width.
         * @param logger Logger used for diagnostics.
         * @param ptr Optional external FP16 destination buffer for preprocessed blob output.
         */
        ImageBatchLoader(
            const fs::path& dirPath,
            size_t batchSize,
            size_t imgH,
            size_t imgW,
            Logger& logger,
            cv::float16_t *ptr = nullptr
        );

        /**
         * @brief Number of batches required to process all images.
         */
        size_t getTotalBatches() const {
            return (m_filesList.size() + m_batchSize - 1) / m_batchSize;
        }

        /**
         * @brief Total number of discovered images.
         */
        size_t getTotalImages() const {
            return m_filesList.size();
        }

        /**
         * @brief Configured batch size.
         */
        size_t getBatchSize() const {
            return m_batchSize;
        }

        /**
         * @brief Ordered list of all discovered image file paths.
         */
        const std::vector<fs::path>& getFileNames() const {
            return m_filesList;
        }

        /**
         * @brief Load and preprocess one batch into the internal/output FP16 tensor.
         * @param batchIdx Zero-based batch index.
         * @param logger Logger used for diagnostics.
         * @param normFactorAddToScaled Normalization add term (currently unused in implementation).
         * @param normFactorScalingMul Normalization multiplier applied in blob creation.
         */
        bool loadBatchDataPreProcessed(
            int batchIdx, 
            Logger& logger, 
            double normFactorAddToScaled, 
            double normFactorScalingMul);

    private:
        std::vector<fs::path> m_filesList;
        size_t m_batchSize, m_imgH, m_imgW;
        ImageBatchData m_batchData;
};