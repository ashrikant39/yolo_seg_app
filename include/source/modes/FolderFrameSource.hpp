#pragma once

#include "source/interface/FrameSource.hpp"
#include "source/config/FrameSourceConfig.hpp"


class FolderFrameSource : public FrameSource {

    public:
        FolderFrameSource(const FrameSourceConfig& config);

        bool read(Frame& frame, BaseLogger& logger) override;

        void reset() {
            m_currId = FRAME_START;
        }

    private:
        fs::path m_folderPath;
        std::vector<fs::path> m_filesList;
        size_t m_currId = FRAME_START;
};