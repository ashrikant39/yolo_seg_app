#include "video.hpp"
#include "utils/options.hpp"
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
ImageBatchLoader::ImageBatchLoader():
    m_filesList({}),
    m_batchSize(0),
    m_imgH(0),
    m_imgW(0),
    m_batchData(0, 0, 0, 0){
    }

// CONSTRUCTOR
ImageBatchLoader::ImageBatchLoader(
    const fs::path& dirPath,
    size_t batchSize,
    size_t imgH,
    size_t imgW,
    Logger& logger,
    cv::float16_t *ptr = nullptr
):
    m_batchSize(batchSize),
    m_imgH(imgH),
    m_imgW(imgW),
    m_batchData(batchSize, imgH, imgW, 3, ptr){
        
        assert(fs::is_directory(dirPath));

        int totalImages = 0;

        for(const auto& entry: fs::directory_iterator(dirPath)){
            if(
                entry.is_regular_file() && (entry.path().extension() == ".png" || entry.path().extension() == ".jpg")){
                m_filesList.push_back(entry.path());
                totalImages++;
            }
        }

        std::sort(m_filesList.begin(), m_filesList.end());

        if(totalImages == 0){
            throw std::runtime_error("No regular files in: " + dirPath.string());
        }

        if(totalImages%batchSize){            
            logger.logConcatMessage(
                Severity::kINFO,
                "Number of files in the directory: ",
                dirPath.c_str(),
                " is: ",
                m_filesList.size(),
                '\n'
            );
        }
    }


void ImageBatchLoader::loadBatchDataPreProcessed(
    int batchIdx,
    Logger& logger,
    double normFactorAddToScaled,
    double normFactorScalingMul){

    size_t totalImgs = m_filesList.size();
    assert(batchIdx * m_batchSize < totalImgs);

    int startIdx = batchIdx * m_batchSize;
    int endIdx = std::min((batchIdx + 1) * m_batchSize, totalImgs);
    
    size_t totalElementsPerImage = m_imgH * m_imgW * 3;
    std::vector<cv::Mat> imageList;
    imageList.reserve(m_batchSize);
    
    try{

        for(int idx = startIdx; idx < startIdx + static_cast<int>(m_batchSize); ++idx){

            if(idx < endIdx){
                cv::Mat image = cv::imread(m_filesList[idx], cv::IMREAD_COLOR);

                if(image.empty()){
                    std::cerr << "Could not read image: " << m_filesList[idx] << '\n';
                    image = cv::Mat::zeros(static_cast<int>(m_imgH), static_cast<int>(m_imgW), CV_8UC3);
                }
                
                imageList.push_back(image);
            } else {
                // Pad partial last batch so blob matches engine batch size
                imageList.push_back(cv::Mat::zeros(static_cast<int>(m_imgH), static_cast<int>(m_imgW), CV_8UC3));
            }
            
        }

        logger.logConcatMessage(
            Severity::kINFO,
            "Loaded Pre-processed images of Batch Index: ",
            batchIdx,
            '\n'
        );
    
        cv::dnn::blobFromImage(
            imageList,
            VideoOptions::NORM_FACTOR_SCALING_MUL,
            cv::Size(m_imgW, m_imgH),
            cv::Scalar(),
            VideoSettings::CHANNEL_ORDER == ChannelOrderMode::RGB,
            false,
            CV_32F
        ).convertTo(m_batchData.images, CV_16F);

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
