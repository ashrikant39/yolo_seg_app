#include <iostream>
#include <filesystem>
#include <string>
#include <cxxopts.hpp>

#include "source/config/FrameSourceConfig.hpp"
#include "backends/config/InferenceBackendConfig.hpp"
#include "logging/BaseLogger.hpp"
#include "memory_management/MemoryManager.hpp"
#include "pre_process/config/PreProcessorConfig.hpp"
#include "post_process/config/PostProcessorConfig.hpp"
#include "sinks/config/ResultSinkConfig.hpp"
#include "AppSettings.hpp"

#include "source/factory/FrameSourceFactory.hpp"
#include "backends/factory/InferenceBackendFactory.hpp"
#include "pre_process/factory/PreProcessorFactory.hpp"
#include "post_process/factory/PostProcessFactory.hpp"
#include "sinks/factory/ResultSinkFactory.hpp"

namespace fs = std::filesystem;

int main(int argc, const char* argv[]){

    try{

        cxxopts::Options options("TestTensorRT", "Program for Running TRT Inference on YOLO-Seg");

        options.add_options()
            ("enginePath", "Engine filepath", cxxopts::value<fs::path>())
            ("logPath", "TRT Log filepath", cxxopts::value<fs::path>()->default_value("example.log"))
            ("videoDirPath", "Directory to load images from.", cxxopts::value<fs::path>())
            ("saveDirPath", "Directory to save predictions to.", cxxopts::value<fs::path>())
            ("saveDetsAsFiles", "Whether to save all the detections per image as a .bin file. If false, no output.", cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
            ("drawMasksOnImage", "Whether to draw the detected masks on the image.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("mode", "Inference mode: folder or video (video not implemented yet).", cxxopts::value<std::string>()->default_value("folder"))
            ("logModelInfo", "Whether to log Model Info", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("h,help", "Print usage");

        auto result = options.parse(argc, argv);

        if(result.count("help")){
            std::cout << options.help() << '\n';
            return 0;
        }
        
        if(!result.count("enginePath")){
            std::cerr << "Engine file not passed! \n";
            return 1;
        }

        if(!result.count("videoDirPath")){
            std::cerr << "videoDirPath not passed! \n";
            return 1;
        }

        if(!result.count("saveDirPath")){
            std::cerr << "saveDirPath not passed! \n";
            return 1;
        }
        
        fs::path logFilePath = result["logPath"].as<fs::path>();
        fs::path engineFileName = result["enginePath"].as<fs::path>();
        fs::path videoDirPath = result["videoDirPath"].as<fs::path>();
        fs::path saveDirPath = result["saveDirPath"].as<fs::path>();
        std::string mode = result["mode"].as<std::string>();
        bool logModelInfo = result["logModelInfo"].as<bool>();
        bool saveDetsAsFile = result["saveDetsAsFiles"].as<bool>();
        bool drawMasksOnImage = result["drawMasksOnImage"].as<bool>();

        if(!fs::exists(engineFileName) || !fs::is_regular_file(engineFileName)){
            std::cerr << "enginePath does not exist or is not a file: " << engineFileName << "\n";
            return 1;
        }

        if(mode == "folder"){
            if(!fs::exists(videoDirPath) || !fs::is_directory(videoDirPath)){
                std::cerr << "videoDirPath does not exist or is not a directory: " << videoDirPath << "\n";
                return 1;
            }
        } else if(mode == "video"){
            std::cerr << "video mode is not implemented yet. Use mode=folder for now.\n";
            return 1;
        } else {
            std::cerr << "Invalid mode: " << mode << ". Expected 'folder' or 'video'.\n";
            return 1;
        }

        // Ensure output directory exists.
        if(fs::exists(saveDirPath)){
            if(!fs::is_directory(saveDirPath)){
                std::cerr << "saveDirPath exists but is not a directory: " << saveDirPath << "\n";
                return 1;
            }
        } else {
            std::error_code ec;
            fs::create_directories(saveDirPath, ec);
            if(ec){
                std::cerr << "Failed to create saveDirPath: " << saveDirPath << " (" << ec.message() << ")\n";
                return 1;
            }
        }

        FrameSourceConfig inputCfg {
            .frameSourceType = Source::FOLDER,
            .sourcePath = videoDirPath,
            .imgHeight = 512,
            .imgWidth = 1024,
            .batchSize = 1
        };

        InferenceBackendConfig inferCfg {
            .inferBackend = BackendType::YoloSegTRT,
            .modelType = ModelType::YOLO_SEGMENTATION,
            .processDevice = ProcessDevice::PREFER_CPU,
            .modelFilePath = engineFileName
        };

        BaseLogger baseLogger(logFilePath);
        MemoryManager manager(
            {
                TensorGroup::PinnedInput,
                TensorGroup::PinnedOutput,
                TensorGroup::HostPostProcessOutput
            }
        );

        PreProcessorConfig preprocessCfg {
            .modelType = ModelType::YOLO_SEGMENTATION,
            .preferredDevice = ProcessDevice::PREFER_CPU,
            .rgbOrdering = ChannelOrder::BGR,
            .resizeHeight = 512,
            .resizeWidth = 1024,
            .n_channels = 3,
            .outputDtype = DType::Float16,
            .scalingFactor = 1.0f/255.0f
        };

        PostProcessorConfig postprocessCfg {
            .outputInfos = {
                {
                    SimplifiedYoloSettings::BOX_KEY,
                    {Shape{{1, 300, 4}}, DType::Float32}
                },

                {
                    SimplifiedYoloSettings::MASK_KEY,
                    {Shape{{1, 300, 128, 256}}, DType::Float32}
                },

                {
                    SimplifiedYoloSettings::CLASS_LABEL,
                    {Shape{{1, 300, 1}}, DType::Float32}
                },

                {
                    SimplifiedYoloSettings::OBJECTNESS,
                    {Shape{{1, 300, 1}}, DType::Float32}
                }

            },
            .modelType = ModelType::YOLO_SEGMENTATION,
            .outputType = OutputType::YOLO_MODIFIED_SEGMENTATION,
            .preferedDevice = ProcessDevice::PREFER_CPU,
            .boxStart = 0,
            .maskStart = 0,
            .classStart = 0,
            .objectnessStart = 0

        };

        ResultSinkConfig resultCfg {
            .sinkMode = ResultSinkMode::SAVE_DETECTIONS,
            .saveMode = SaveDetectionMode::NORMALIZED,
        };

        std::unique_ptr<FrameSource> source = createFrameSource(inputCfg);
        std::unique_ptr<PreProcessor> preprocessor = createPreProcessor(preprocessCfg);
        std::unique_ptr<InferenceBackend> infer = createInferenceBackend(inferCfg, baseLogger);
        std::unique_ptr<PostProcessor> postprocessor = createPostProcessor(postprocessCfg);
        std::unique_ptr<ResultSink> sink = createResultSink(config);

        

        return 0;
    }
    catch(const cxxopts::exceptions::exception& e) {
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }
    catch(const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
