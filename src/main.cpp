#include <iostream>
#include "pipeline.hpp"
#include <cxxopts.hpp>

int main(int argc, const char* argv[]){

    try{

        cxxopts::Options options("TestTensorRT", "Program for Running TRT Inference on YOLO-Seg");

        options.add_options()
            ("enginePath", "Engine filepath", cxxopts::value<fs::path>())
            ("logPath", "TRT Log filepath", cxxopts::value<fs::path>()->default_value("example.log"))
            ("videoDirPath", "Directory to load images from.", cxxopts::value<fs::path>())
            ("saveDirPath", "Directory to save predictions to.", cxxopts::value<fs::path>())
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
            std::cerr << "Image Path not passed! \n";
            return 1;
        }

        if(!result.count("saveDirPath")){
            std::cerr << "Image Path not passed! \n";
            return 1;
        }
        
        fs::path logFilePath = result["logPath"].as<fs::path>();
        fs::path engineFileName = result["enginePath"].as<fs::path>();
        fs::path videoDirPath = result["videoDirPath"].as<fs::path>();
        fs::path saveDirPath = result["saveDirPath"].as<fs::path>();
        bool logModelInfo = result["logModelInfo"].as<bool>();
        
        InferencePipeline pipeline(engineFileName, logFilePath, videoDirPath, saveDirPath, logModelInfo);
        pipeline.runInferencePipeline();
        
        return 0;
    }
    catch(const cxxopts::exceptions::exception& e){
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }

}
