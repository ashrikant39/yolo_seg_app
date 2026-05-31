#pragma once

#include <opencv2/videoio.hpp> 
#include "source/interface/FrameSource.hpp"
#include "source/config/FrameSourceConfig.hpp"

class VideoFrameSource : public FrameSource {

    public:
        VideoFrameSource(const FrameSourceConfig& config);

        bool read(Frame& frame, Logger& logger) override;

        double getFPS() {
            return cap.get(cv::CAP_PROP_FPS);
        }

        size_t getTotalFrames() {
            return cap.get(cv::CAP_PROP_FRAME_COUNT);
        }
    
    private:
        fs::path m_videoPath;
        cv::VideoCapture m_cap;
};