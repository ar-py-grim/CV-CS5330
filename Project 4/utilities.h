/*
Arpit Gandhi
March 2026

Include file for utilities.cpp and ar_gl.cpp file.
Contians function prototypes for aruco board creation, corner detection, camera calibration,
pose estimation, AR overlay, feature detection, and AR with OpenGL.
*/

#ifndef UTILITIES_H
#define UTILITIES_H

int charucoboardCreate();
int detectCorners();
int calibrateCamera();
int poseEstimation();
int arOverlay();
int featureDetection();
int arGL();

#endif