#pragma once

#include <opencv2/opencv.hpp>
#include <variant>
#include <string>
#include <vector>

void preprocessSingleFrame(const cv::Mat&, cv::Mat&, cv::Size);

void preprocessBatch(const std::vector<cv::Mat>&, std::vector<cv::Mat>&, cv::Size);
