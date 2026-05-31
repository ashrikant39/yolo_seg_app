#include <opencv2/imgcodecs.hpp>

#include "sinks/modes/DrawDetectionSink.hpp"
#include "core/cuda.hpp"
#include "post_process/utils/MatUtils.hpp"
#include "sinks/utils/drawUtils.hpp"


// CONSTRUCTOR
DrawDetectionSink::DrawDetectionSink(const ResultSinkConfig& config):
    m_lineThickness(config.lineThickness) {

    // switch (config.drawMaskMode) {

    //     case MaskDrawingMode::FROM_CONTOUR:
    //         m_drawContourBasedMasks = true;
    //         break;

    //     case  MaskDrawingMode::RAW:
    //         m_drawContourBasedMasks = false;
    //         break;

    //     case MaskDrawingMode::UNSET:
    //         m_drawContourBasedMasks = true;
    //         break;
        
    //     default:
    //         throw std::runtime_error("Invalid MaskDrawingMode Parameter ");
    // }

    switch (config.drawDetectionMode)
    {
        case DrawDetectionMode::BOXES_ONLY:
            m_drawBoxes = true;
            m_drawContours = false;
            m_drawMasks = false;
            break;

        case DrawDetectionMode::CONTOURS_ONLY:
            m_drawBoxes = false;
            m_drawContours = true;
            m_drawMasks = false;
            break;
            
        case DrawDetectionMode::MASKS_ONLY:
            m_drawBoxes = false;
            m_drawContours = false;
            m_drawMasks = true;
            break;
            
        case DrawDetectionMode::COUNTOURS_WITH_BOXES:
            m_drawBoxes = true;
            m_drawContours = true;
            m_drawMasks = false;
            break;
            
        case DrawDetectionMode::MASKS_WITH_BOXES:
            m_drawBoxes = true;
            m_drawContours = false;
            m_drawMasks = true;
            break;

        case DrawDetectionMode::UNSET:
            m_drawBoxes = true;
            m_drawContours = false;
            m_drawMasks = false;
            break;
            
        default:
            throw std::runtime_error("Invalid DrawDetectionMode Parameter");
    }

}

void DrawDetectionSink::consumeSingle(PostProcessOutput& output, Logger& logger) {


    if ( output.metadata.imagePath.empty() || !std::filesystem::is_regular_file(output.metadata.imagePath) ) {
        throw std::runtime_error("Image Path not found: " + output.metadata.imagePath.string());
    }

    if ( output.metadata.resultsDir.empty() || !std::filesystem::is_directory(output.metadata.resultsDir) ) {
        throw std::runtime_error("Cannot Save results to: " + output.metadata.resultsDir.string() );
    }

    cv::Mat resizedImg;
    size_t imageW = output.metadata.inputWidth;
    size_t imageH = output.metadata.inputHeight;

    cv::resize(cv::imread(output.metadata.imagePath, cv::IMREAD_COLOR), resizedImg, cv::Size(imageW, imageH));
    std::string dirName = output.metadata.saveMaskDirName;


    if ( dirName.empty() ) {
        dirName = "drawDetections"; 
    }

    fs::path saveDir = output.metadata.resultsDir / dirName;
                      
    fs::create_directories(saveDir);
    size_t idx = 0;

    for (const auto& detection : output.detections ) {
        fs::path maskPath = saveDir / 
                            detection.metadata.imgPath.stem() /
                            "_drawn_detections_" / 
                            std::to_string(detection.metadata.detectionId) /
                            ".png";
        
        // contour = denormalizeToIntContour()
        // drawDetectedMasksOnImage()
        idx++;
    }

}