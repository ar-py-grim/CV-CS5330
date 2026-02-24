/*
Arpit Gandhi
February 2026

Include file for utilities.cpp file.
Contians image embeddings function prototypes.
*/

#ifndef UTILITIES_H
#define UTILITIES_H

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

using namespace cv;
using namespace std;

int getEmbedding(Mat& src, Mat& embedding, dnn::Net& net, int debug);
void prepEmbeddingImage(Mat& frame, Mat& embimage, int cx, int cy, float theta, float minE1,
    float maxE1, float minE2, float maxE2, int debug);

#endif
