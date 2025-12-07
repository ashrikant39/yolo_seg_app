#include "video.hpp"
#include "options.hpp"
#include "settings.hpp"
#include "utils/enums.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <opencv2/dnn.hpp>
#include <nvToolsExt.h>
#include <vector>
#include "settings.hpp"

#define NVTX_RANGE(name) do { nvtxRangePushA(name); } while (0)
#define NVTX_POP()      do { nvtxRangePop(); } while (0)


// DEF-CONSTRUCTOR
VideoFromDirectory::VideoFromDirectory():
    _filesList({}),
    _batchSize(0),
    _imgH(0),
    _imgW(0),
    _batchData(0, 0, 0, 0){
    }

// CONSTRUCTOR
VideoFromDirectory::VideoFromDirectory(
    const fs::path& dirPath,
    size_t batchSize,
    size_t imgH,
    size_t imgW,
    Logger& logger
):
    _batchSize(batchSize),
    _imgH(imgH),
    _imgW(imgW),
    _batchData(batchSize, imgH, imgW, 3){
        
        assert(fs::is_directory(dirPath));

        int totalImages = 0;

        for(const auto& entry: fs::directory_iterator(dirPath)){
            if(
                entry.is_regular_file() && (entry.path().extension() == ".png" || entry.path().extension() == ".jpg")){
                _filesList.push_back(entry.path());
                totalImages++;
            }
        }

        std::sort(_filesList.begin(), _filesList.end());

        if(totalImages == 0){
            throw std::runtime_error("No regular files in: " + dirPath.string());
        }

        if(totalImages%batchSize){            
            logger.logConcatMessage(
                Severity::kINFO,
                "Number of files in the directory: ",
                dirPath.c_str(),
                " is: ",
                _filesList.size(),
                '\n'
            );
        }
    }


const ImageBatchData& VideoFromDirectory::getBatchDataPreProcessed(
    int batchIdx,
    Logger& logger,
    double normFactorAddToScaled,
    double normFactorScalingMul){

    size_t totalImgs = _filesList.size();
    assert(batchIdx*_batchSize < totalImgs);

    int startIdx = batchIdx * _batchSize;
    int endIdx = std::min((batchIdx + 1)*_batchSize, totalImgs);
    
    size_t totalElementsPerImage = _imgH*_imgW*3;
    std::vector<cv::Mat> images;
    images.reserve(_batchSize);
    
    try{

        for(int idx=startIdx; idx<startIdx+_batchSize; ++idx){

            if(idx < endIdx){
                cv::Mat image = cv::imread(_filesList[idx], cv::IMREAD_COLOR);

                if(image.empty()){
                    std::cerr << "Could not read image: " << _filesList[idx] << '\n';
                }
                
                images.push_back(image);
            }
            
        }

        logger.logConcatMessage(
            Severity::kINFO,
            "Loaded Pre-processed images of Batch Index: ",
            batchIdx,
            '\n'
        );
    
        cv::dnn::blobFromImage(
            images,
            VideoOptions::NORM_FACTOR_SCALING_MUL,
            cv::Size(_imgW, _imgH),
            cv::Scalar(),
            VideoSettings::CHANNEL_ORDER == ChannelOrderMode::RGB,
            false,
            CV_32F
        ).convertTo(_batchData.images, CV_16F);

        return _batchData;
    }
    
    catch(const cv::Exception& e){
        std::cerr << "OpenCV error: " << e.what() << "\n";
        std::cerr << "Code: " << e.code << "\n";
        std::cerr << "Func: " << e.func << "\n";
        std::cerr << "File: " << e.file << "\n";
        std::cerr << "Line: " << e.line << "\n";
    }

    catch(std::exception& e){

        logger.logConcatMessage(
            Severity::kINFO,
            "Exception: ",
            e.what(),
            "\nWhile loading Batch Index: ",
            batchIdx,
            '\n'
        );
        throw e;
    }
}
