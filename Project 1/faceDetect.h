/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing functions
*/

#ifndef FACEDETECT_H
#define FACEDETECT_H
using namespace std;
using namespace cv;

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "./haarcascade_frontalface_alt2.xml"

// prototypes
int detectFaces(Mat &grey, vector<Rect> &faces);
int drawBoxes(Mat &frame, vector<Rect> &faces, int minWidth = 50, float scale = 1.0);

#endif
