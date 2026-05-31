#include "source/modes/FolderFrameSource.hpp"


FolderFrameSource::FolderFrameSource(const FrameSourceConfig& config):
    m_imgHeight(config.imgHeight),
    m_imgWidth(config.imgWidth),
    m_batchSize(config.batchSize),
    m_folderPath(config.frameSource) {

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
            }

        }

        if(totalImages == 0){
            throw std::runtime_error("No image files in: " + m_folderPath.string());
        }

        std::sort(m_filesList.begin(), m_filesList.end());

}

bool FolderFrameSource::read(Frame& frame, Logger& logger) {


    if ( m_currId >= m_filesList.size() ) {
        if ( m_currId > ((m_filesList.size() + m_batchSize - 1)/m_batchSize) * m_batchSize ) {
            return false;
        }

        frame.image = zeros();
        frame.metadata.frameId = m_currId;
        frame.metadata.sourcePath = fs::path("");
        frame.metadata.timeStampNs = INVALID_TIMESTAMP;

        return true;
    }
    

    fs::path imgPath = m_filesList[m_currId];

    frame.metadata.frameId = m_currId;
    frame.metadata.sourcePath = imgPath;
    frame.metadata.timeStamp = INVALID_TIMESTAMP;

    frame.image = cv::imread(imgPath, cv::IMREAD_COLOR);

    if (frame.image.empty()) {
        logger.log(Severity::kWARNING, 
            "Could not read image: ", 
            imgPath.string(), 
            "Using Zeros."
        );
        frame.image = zeros();
    }

    m_currId++;

    return true;
}