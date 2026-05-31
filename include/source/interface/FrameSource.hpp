#pragma once

#include "source/utils/frame.hpp"
#include "core/logger.hpp"


class FrameSource {

    public:

        virtual ~FrameSource() = default;
        virtual bool read(Frame& frame, Logger& logger) = 0;
        
        bool readBatch(BatchFrameData& batch, Logger& logger) {

            batch.images.reserve(m_batchSize);
            batch.metas.reserve(m_batchSize);
            
            Frame tmpFrame;

            for (int i = 0; i < m_batchSize; i++) {
            
                if (!read(tmpFrame, logger)) {
                    return false;
                }

                batch.images.push_back(tmpFrame.image);
                batch.metas.push_back(tmpFrame.metadata);

            }

            return true;
        }

    protected:
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