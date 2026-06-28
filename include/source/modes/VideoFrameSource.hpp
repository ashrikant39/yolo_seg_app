#pragma once

#include <opencv2/videoio.hpp> 
#include "source/interface/FrameSource.hpp"
#include "source/config/FrameSourceConfig.hpp"

/**
 * @brief FrameSource implementation backed by cv::VideoCapture.
 */
class VideoFrameSource : public FrameSource {

    public:
        /**
         * @brief Construct a video source.
         * @param config Video path, frame dimensions, and batch size.
         */
        VideoFrameSource(const FrameSourceConfig& config);

        /**
         * @copydoc FrameSource::read
         */
        bool read(Frame& frame, BaseLogger& logger) override;

        /**
         * @brief Query source video FPS from OpenCV.
         * @return Frames per second reported by cv::VideoCapture.
         */
        double getFPS() {
            return m_cap.get(cv::CAP_PROP_FPS);
        }

        /**
         * @brief Query total frame count from OpenCV.
         * @return Number of frames reported by cv::VideoCapture.
         */
        size_t getTotalFrames() {
            return m_cap.get(cv::CAP_PROP_FRAME_COUNT);
        }
    
    private:
        fs::path m_videoPath;
        cv::VideoCapture m_cap;
        size_t m_paddingFrameId = 0;
};
