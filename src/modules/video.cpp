#include "video.h"
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <cstring>

void preprocessImage(
    const cv::Mat& img, 
    cv::Mat& result, 
    size_t imgH, 
    size_t imgW, 
    double normFactorAddToScaled, 
    double normFactorScalingMul){
    
    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size(imgW, imgH));    
    resizedImage.convertTo(result, CV_16F, normFactorScalingMul, normFactorAddToScaled);
       
}

// CONSTRUCTOR
VideoFromDirectory::VideoFromDirectory(
    const fs::path& dirPath,
    size_t batchSize,
    size_t imgH,
    size_t imgW,
    Logger& logger
):
    m_batchSize(batchSize),
    m_imgH(imgH),
    m_imgW(imgW),
    m_batchData(batchSize, imgH*imgW*3){

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
            std::stringstream ss;
            ss << "Number of files in the directory: " << dirPath.c_str() << " is: " << m_filesList.size();
            logger.log(Severity::kWARNING, ss.str().c_str());
        }
    }


const ImageBatchData& VideoFromDirectory::getBatchDataPreProcessed(
    int batchIdx, 
    Logger& logger, 
    double normFactorAddToScaled, 
    double normFactorScalingMul){

    size_t totalImgs = m_filesList.size();
    assert(batchIdx*m_batchSize < totalImgs);

    int startIdx = batchIdx * m_batchSize;
    int endIdx = std::min((batchIdx + 1)*m_batchSize, totalImgs);
    
    cv::Mat image, preProcessedImage;
    size_t totalElementsPerImage = m_imgH*m_imgW*3;
    
    try{

        for(int idx=startIdx; idx<startIdx+m_batchSize; ++idx){

            if(idx < endIdx){
                image = cv::imread(m_filesList[idx], cv::IMREAD_COLOR);
                m_batchData.filePaths.push_back(m_filesList[idx]);
            }
            else{
                image = cv::Mat::zeros(m_imgH, m_imgW, CV_8UC3);
            }

            preprocessImage(image, preProcessedImage, m_imgH, m_imgW, normFactorAddToScaled, normFactorScalingMul);            
            
            std::memcpy(
                &m_batchData.dataBuffer[(idx - startIdx) * totalElementsPerImage],
                preProcessedImage.ptr(),
                totalElementsPerImage * sizeof(cv::float16_t)
            );
        }
        
        std::stringstream ss;
        ss << "Loaded Pre-processed images of Batch Index: " << batchIdx;
        logger.log(Severity::kINFO, ss.str().c_str());

        return m_batchData;
    }
    catch(std::exception& e){

        std::stringstream ss;
        ss << "Exception: " << e.what() << "\nWhile loading Batch Index: " << batchIdx;
        logger.log(Severity::kERROR, ss.str().c_str());
        throw e;
    }
    
}
