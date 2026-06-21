#include <algorithm>
#include <string>

#include "source/modes/FolderFrameSource.hpp"
#include "AppSettings.hpp"


FolderFrameSource::FolderFrameSource(const FrameSourceConfig& config):
    FrameSource(config.imgHeight, config.imgWidth, config.batchSize),
    m_folderPath(config.sourcePath) {

        if (!fs::is_directory(m_folderPath)) {
            throw std::runtime_error("Invalid source path for a folder : " + m_folderPath.string());
        }


        int totalImages = 0;

        for(const auto& entry: fs::directory_iterator(m_folderPath)){

            if (!entry.is_regular_file()) {
                continue;
            }

            const auto ext = entry.path().extension();

            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                m_filesList.push_back(entry.path());
                totalImages++;
            }

        }

        if(totalImages == 0){
            throw std::runtime_error("No image files in: " + m_folderPath.string());
        }

        std::sort(m_filesList.begin(), m_filesList.end());

}

bool FolderFrameSource::read(Frame& frame, BaseLogger& logger) {

    if ( m_currId >= m_filesList.size() ) {
        const size_t paddedSize = ((m_filesList.size() + m_batchSize - 1) / m_batchSize) * m_batchSize;
        if (m_currId >= paddedSize) {
            return false;
        }

        frame.image = zeros();
        frame.metadata.frameId = m_currId;
        frame.metadata.imagePath = fs::path("");
        frame.metadata.timestampNs = INVALID_TIMESTAMP;
        frame.metadata.originalWidth = m_imgWidth;
        frame.metadata.originalHeight = m_imgHeight;
        frame.metadata.originalChannels = StaticSettings::NUM_IMG_CHANNELS;
        frame.metadata.isPadding = true;

        ++m_currId;

        return true;
    }
    

    fs::path imgPath = m_filesList[m_currId];

    frame.metadata.frameId = m_currId;
    frame.metadata.sourcePath = m_folderPath;
    frame.metadata.imagePath = imgPath;
    frame.metadata.timestampNs = INVALID_TIMESTAMP;
    frame.metadata.originalWidth = m_imgWidth;
    frame.metadata.originalHeight = m_imgHeight;
    frame.metadata.originalChannels = StaticSettings::NUM_IMG_CHANNELS;
    frame.metadata.isPadding = false;

    const std::string imgPathString = imgPath.string();
    frame.image = cv::imread(imgPathString.c_str(), cv::IMREAD_COLOR);

    if (frame.image.empty()) {
        logger.logConcatMessage(
            LoggingSeverityType::ERROR,
            "Could not read image: ",
            imgPath.string(),
            ". Using zeros.\n"
        );
        frame.image = zeros();
    }

    m_currId++;

    return true;
}
