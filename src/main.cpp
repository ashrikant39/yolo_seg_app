#include <filesystem>
#include <iostream>

#include <cxxopts.hpp>

#include "application/Application.hpp"

namespace fs = std::filesystem;

int main(int argc, const char* argv[]) {
    try {
        cxxopts::Options options(
            "yoloSegApp",
            "Run YOLO segmentation inference from a YAML configuration file"
        );

        options.add_options()
            ("c,config", "YAML configuration file", cxxopts::value<fs::path>())
            ("h,help", "Print usage");
        options.parse_positional({"config"});
        options.positional_help("<config.yaml>");

        const auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << '\n';
            return 0;
        }

        if (!result.count("config")) {
            std::cerr << "Missing YAML configuration file.\n\n";
            std::cerr << options.help() << '\n';
            return 1;
        }

        const fs::path configPath = result["config"].as<fs::path>();
        if (!fs::is_regular_file(configPath)) {
            std::cerr << "Config path does not exist or is not a file: " << configPath << '\n';
            return 1;
        }

        Application app(configPath);
        app.run();

    } catch (const cxxopts::exceptions::exception& e) {
        
        std::cerr << "Error parsing options: " << e.what() << '\n';
        return 1;
        
    } catch (const std::exception& e) {

        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
        
    }
}
