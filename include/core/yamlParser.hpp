#pragma once

#include <filesystem>

#include "AppSettings.hpp"

namespace fs = std::filesystem;

/**
 * @brief Parse application settings from a YAML configuration file.
 * @param yamlPath YAML file path.
 * @return Parsed AppSettings used to construct Application.
 * @throws std::runtime_error if required keys are missing or values are invalid.
 */
AppSettings loadAppSettingsFromYaml(const fs::path& yamlPath);
