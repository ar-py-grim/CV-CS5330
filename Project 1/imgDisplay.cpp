/*
Arpit Gandhi
 January 2025

This file just loads and displays an image using OpenCV.

*/ 

#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // load image
    Mat image = imread("C:/Users/ASUS/Desktop/CS5330/Project 1/images/scenery.jpg");
	// some image properties
    cout << "No.of channels = " << image.channels() << endl;
    cout << "No. of rows= " << image.rows <<" No. of clms= " << image.cols << endl;
	cout << "Image Type = " << image.type() << endl;
	// check if image is loaded successfully
    if (image.empty())
    {
        cout << "Image File Not Found" << endl;
        return -1;
    }
    cv::imshow("Original", image);
	// wait for a key press indefinitely
    if ((waitKey(0) & 0xFF) == 'q') 
    {
        destroyAllWindows();
    }
    return 0;
}