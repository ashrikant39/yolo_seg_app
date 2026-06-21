#include <string>

#include "source/modes/VideoFrameSource.hpp"
#include "AppSettings.hpp"

VideoFrameSource::VideoFrameSource(const FrameSourceConfig& config):
    FrameSource(config.imgHeight, config.imgWidth, config.batchSize),
    m_videoPath(config.sourcePath) {

        const std::string videoPathString = m_videoPath.string();
        m_cap.open(videoPathString.c_str());

        if (!m_cap.isOpened()) {
            throw std::runtime_error("Video could not be opened.");
        }
}


bool VideoFrameSource::read(Frame& frame, BaseLogger& logger) {

    size_t currFrameId = static_cast<uint64_t>(m_cap.get(cv::CAP_PROP_POS_FRAMES));

    frame.metadata.sourcePath = m_videoPath;
    frame.metadata.imagePath = m_videoPath.stem().string() + ("_frame_" + std::to_string(currFrameId) + ".jpg");
    frame.metadata.frameId = currFrameId;
    double fps = getFPS();
    frame.metadata.timestampNs = static_cast<uint64_t>((1e9 * frame.metadata.frameId) / fps);
    frame.metadata.isPadding = false;
    frame.metadata.originalWidth = m_imgWidth;
    frame.metadata.originalHeight = m_imgHeight;
    frame.metadata.originalChannels = StaticSettings::NUM_IMG_CHANNELS;

    size_t totalFrames = getTotalFrames();

    if ( currFrameId >= totalFrames ) {
        const size_t paddedSize = ((totalFrames + m_batchSize - 1) / m_batchSize) * m_batchSize;
        if (m_paddingFrameId == 0) {
            m_paddingFrameId = currFrameId;
        }
        if (m_paddingFrameId >= paddedSize) {
            return false;
        }
        frame.image = zeros();
        frame.metadata.frameId = m_paddingFrameId++;
        frame.metadata.isPadding = true;
        return true;
    }

    return m_cap.read(frame.image);
}
