#include <iostream>
#include <filesystem>
#include <string>
#include "pipeline.hpp"
#include <cxxopts.hpp>

namespace fs = std::filesystem;

int main(int argc, const char* argv[]){

    try{

        cxxopts::Options options("TestTensorRT", "Program for Running TRT Inference on YOLO-Seg");

        options.add_options()
            ("enginePath", "Engine filepath", cxxopts::value<fs::path>())
            ("logPath", "TRT Log filepath", cxxopts::value<fs::path>()->default_value("example.log"))
            ("videoDirPath", "Directory to load images from.", cxxopts::value<fs::path>())
            ("saveDirPath", "Directory to save predictions to.", cxxopts::value<fs::path>())
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
        
        InferencePipeline pipeline(engineFileName, logFilePath, videoDirPath, saveDirPath, logModelInfo);
        pipeline.runInferencePipeline();
        
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
