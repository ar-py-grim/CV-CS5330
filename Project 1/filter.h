/*
Arpit Gandhi
 January 2025

Include file for filter.cpp file.
Contians various image filter function prototypes.
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class DA2Network;

int color_to_gray(Mat &src, Mat &dst);
int sepia_filter(Mat &src, Mat &dst, int levels=2);
int blur5x5_1(Mat &src, Mat &dst);
int blur5x5_2(Mat &src, Mat &dst);
int sobelX3x3(Mat &src, Mat &dst);
int sobelY3x3(Mat &src, Mat &dst);
int magnitude(Mat &sx, Mat &sy, Mat &dst);
int blurQuantize(Mat &src, Mat &dst, int levels=10);
int embossing(Mat &src, Mat &dst);
int faceBlur(Mat &frame, vector<Rect> &faces, int ksize=11, float scale=1.0, double sigx=1.0, double sigy=1.0);
int fogEffect(Mat &src, Mat &dst, DA2Network &depthNet, int fog_intensity=200, float density=2.0);
int faceHiglight(Mat &frame, vector<Rect> &faces, float scale = 1.0);
int cannyEdge(Mat &src, Mat &dst, int threshold=100);

#endif
