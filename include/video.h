#include <opencv2/opencv.hpp>
#include <variant>

using videoSourceType = std::variant<std::string, int>;

class VideoProcessor
{
    public:
        
        VideoProcessor();
        VideoProcessor(const videoSourceType& videoPath, const int batchSize)

    private:
        
        videoSourceType videoPath;
        const int batchSize;
};