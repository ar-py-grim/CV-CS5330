/*
Arpit Gandhi
February 2025

Feature extraction and distance metrics prototypes

*/

#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <opencv2/opencv.hpp>


// extracts a 7x7 pixel region from the center of the image and returns it
std::vector<float> extract_center(cv::Mat& image);

// computes and returns ssd distance
float ssd(std::vector<float>& f1, std::vector<float>& f2);

// 2D RG Histogram
std::vector<float> rg_histogram(cv::Mat& image, int histsize=16);

// 3D RGB Histogram
std::vector<float> rgb_histogram(cv::Mat& image, int histsize=8);

// histogram intersection distance
float hist_inter(std::vector<float>& h1, std::vector<float>& h2);

int sobelX3x3(cv::Mat& src, cv::Mat& dst);

int sobelY3x3(cv::Mat& src, cv::Mat& dst);

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

std::vector<float> texture_histogram(cv::Mat& image, int histsize=8);

std::vector<float> gabor_features(cv::Mat& image, int gsize=15, int num_theta=4, int num_scales=2);

#endif