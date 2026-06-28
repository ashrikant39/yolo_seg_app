#pragma once

#include "source/interface/FrameSource.hpp"
#include "source/config/FrameSourceConfig.hpp"


/**
 * @brief FrameSource implementation that reads images from a folder.
 *
 * Supported file extensions are discovered in the constructor and read in
 * sorted path order. Final incomplete batches are padded with zero frames.
 */
class FolderFrameSource : public FrameSource {

    public:
        /**
         * @brief Construct a folder source.
         * @param config Source path, frame dimensions, and batch size.
         */
        FolderFrameSource(const FrameSourceConfig& config);

        /**
         * @copydoc FrameSource::read
         */
        bool read(Frame& frame, BaseLogger& logger) override;

        /**
         * @brief Reset iteration to the first frame.
         */
        void reset() {
            m_currId = FRAME_START;
        }

    private:
        fs::path m_folderPath;
        std::vector<fs::path> m_filesList;
        size_t m_currId = FRAME_START;
};
