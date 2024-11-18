#include <video.h>


void preprocessSingleFrame(const cv::Mat& inputFrame, cv::Mat& processedFrame, cv::Size size)
{
    cv::Mat resizedFrame;
    cv::resize(inputFrame, resizedFrame, size);
    resizedFrame.convertTo(processedFrame, CV_32FC3, 1./255);
}

void preprocessBatch(const std::vector<cv::Mat>& inputFrameBatch, std::vector<cv::Mat>& processedFrameBatch, cv::Size finalSize)
{
    for(int i=0; i<inputFrameBatch.size(); i++)
    {
        cv::Mat outputFrame;
        preprocessSingleFrame(inputFrameBatch[i], outputFrame, finalSize);
        processedFrameBatch.emplace_back(outputFrame);
    }
}



