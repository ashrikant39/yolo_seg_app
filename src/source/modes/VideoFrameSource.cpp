#pragma once

#include "source/modes/VideoFrameSource.hpp"

VideoFrameSource::VideoFrameSource(const FrameSourceConfig& config):
    m_imgHeight(config.imgHeight),
    m_imgWidth(config.imgWidth),
    m_batchSize(config.batchSize),
    m_videoPath(config.sourcePath) {

        m_cap.open(m_videoPath);

        if (!m_cap.opened()) {
            throw std::runtime_error("Video could not be opened.");
        }
}


bool VideoFrameSource::read(Frame& frame, Logger& logger) {

    size_t currFrameId = static_cast<uint64_t>(m_cap.get(cv::CAP_PROP_POS_FRAMES));

    frame.metadata.sourcePath = m_videoPath;
    frame.metadata.frameId = currFrameId;
    double fps = getFPS();
    frame.metadata.timeStampNs = static_cast<uint64_t>((1e9 * frame.metadata.frameId) / fps);
    
    size_t totalFrames = getTotalFrames();

    if ( currFrameId >= totalFrames ) {
        if ( currFrameId > (( totalFrames + m_batchSize - 1)/m_batchSize) * m_batchSize ) {
            return false;
        }
        frame.image = zeros();
        return true;
    }

    return m_cap.read(frame.image);
}