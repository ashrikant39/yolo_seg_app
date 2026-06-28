#pragma once

#include "source/utils/frame.hpp"
#include "logging/BaseLogger.hpp"


/**
 * @brief Abstract source of frames for the pipeline.
 *
 * Derived classes implement read() for a single frame. The base class provides
 * readBatch() and zero-frame padding support for full-batch execution.
 */
class FrameSource {

    public:

        virtual ~FrameSource() = default;

        /**
         * @brief Read one frame from the source.
         * @param frame Destination frame and metadata.
         * @param logger Logger for source diagnostics.
         * @return true when a frame or padding frame was produced; false at end of source.
         */
        virtual bool read(Frame& frame, BaseLogger& logger) = 0;
        
        /**
         * @brief Read a full batch of frames.
         * @param batch Destination batch; cleared before reading.
         * @param logger Logger for source diagnostics.
         * @return true if a full batch was read; false if the source ended early.
         */
        bool readBatch(BatchFrameData& batch, BaseLogger& logger) {

            batch.images.clear();
            batch.metas.clear();
            batch.images.reserve(m_batchSize);
            batch.metas.reserve(m_batchSize);
            
            Frame tmpFrame;

            for (size_t i = 0; i < m_batchSize; i++) {
            
                if (!read(tmpFrame, logger)) {
                    return false;
                }

                batch.images.push_back(tmpFrame.image);
                batch.metas.push_back(tmpFrame.metadata);

            }

            return true;
        }

    protected:
        /**
         * @brief Construct shared source state.
         * @param imgHeight Frame height used for zero padding frames.
         * @param imgWidth Frame width used for zero padding frames.
         * @param batchSize Number of frames per batch.
         */
        FrameSource(size_t imgHeight, size_t imgWidth, size_t batchSize):
            m_imgHeight(imgHeight),
            m_imgWidth(imgWidth),
            m_batchSize(batchSize) {}

        /**
         * @brief Create a black frame with configured source dimensions.
         * @return CV_8UC3 zero image.
         */
        cv::Mat zeros() {
            return cv::Mat::zeros(
                static_cast<int>(m_imgHeight),
                static_cast<int>(m_imgWidth),
                CV_8UC3
            );
        }
    
    protected:
        size_t m_imgHeight, m_imgWidth, m_batchSize;
};
