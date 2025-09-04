#include "video.h"
#include "options.h"
#include "settings.h"
#include "types/enums.h"
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
#include "settings.h"
#include <thread>

#define NVTX_RANGE(name) do { nvtxRangePushA(name); } while (0)
#define NVTX_POP()      do { nvtxRangePop(); } while (0)


void preprocessImage(
    const cv::Mat& img,
    cv::Mat& result,
    size_t imgH,
    size_t imgW,
    double normFactorAddToScaled,
    double normFactorScalingMul){

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat resizedImage;
        cv::resize(img, resizedImage, cv::Size(imgW, imgH));    
        resizedImage.convertTo(result, CV_16F, normFactorScalingMul, normFactorAddToScaled);
}

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
        
        //cv::setUseOptimized(true);
        //cv::setNumThreads(std::thread::hardware_concurrency());

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
    
    cv::Mat image, preProcessedImage;
    size_t totalElementsPerImage = _imgH*_imgW*3;
    
    try{

        for(int idx=startIdx; idx<startIdx+_batchSize; ++idx){

            if(idx < endIdx){
                image = cv::imread(_filesList[idx], cv::IMREAD_COLOR);

                if(image.empty()){
                    std::cerr << "Could not read image: " << _filesList[idx] << '\n';
                }
               
                NVTX_RANGE("blobFromImage");
                preProcessedImage = cv::dnn::blobFromImage(
                        image,
                        VideoOptions::NORM_FACTOR_SCALING_MUL,
                        cv::Size(_imgW, _imgH),
                        cv::Scalar(),
                        VideoSettings::CHANNEL_ORDER == ChannelOrderMode::BGR,
                        false,
                        CV_32F
                );
                NVTX_POP();
                
                NVTX_RANGE("MemCpyImages");
                std::memcpy(
                        _batchData.images.ptr<cv::float16_t>(idx - startIdx),
                        preProcessedImage.ptr<cv::float16_t>(0),
                        totalElementsPerImage
                        );
                NVTX_POP();
            }
            
        }

        logger.logConcatMessage(
            Severity::kINFO,
            "Loaded Pre-processed images of Batch Index: ",
            batchIdx,
            '\n'
        );
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
