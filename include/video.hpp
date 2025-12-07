#pragma once

#include <filesystem>
#include <vector>
#include "logger.hpp"
#include "utils/input.hpp"

namespace fs = std::filesystem;

class VideoFromDirectory{

    public: 

        VideoFromDirectory();

        VideoFromDirectory(
            const fs::path& dirPath,
            size_t batchSize,
            size_t imgH,
            size_t imgW,
            Logger& logger
        );

        size_t getTotalBatches(){
            return (_filesList.size() + _batchSize - 1)/_batchSize;
        }

        size_t getTotalImages(){
            return _filesList.size();
        }

        const ImageBatchData& getBatchDataPreProcessed(
            int batchIdx, 
            Logger& logger, 
            double normFactorAddToScaled, 
            double normFactorScalingMul);

    private:
        std::vector<fs::path> _filesList;
        size_t _batchSize, _imgH, _imgW;
        ImageBatchData _batchData;
};