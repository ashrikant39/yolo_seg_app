#pragma once

#include <filesystem>

#include "AppSettings.hpp"

namespace fs = std::filesystem;

AppSettings loadAppSettingsFromYaml(const fs::path& yamlPath);

