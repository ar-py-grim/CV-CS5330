/*
Arpit Gandhi
March 2026

Main file to run different functions related to ChArUco board creation, corner detection, camera calibration,
pose estimation, AR overlay, feature detection, and AR with OpenGL. 
User can select which function to run by entering a number from 0 to 6.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp> 
#include <iostream>
#include "utilities.h"

using namespace cv;
using namespace std;

int main()
{
    int choice;
    cout << "0: Generate ChArUco board" << endl;
    cout << "1: Detect ChArUco corners" << endl;
    cout << "2: Camera Caliberation" << endl;
    cout << "3: Pose Estimation" << endl;
    cout << "4: AR Overlay" << endl;
    cout << "5: Feature Dectection" << endl;
    cout << "6: Add virtual object in scene" << endl;
    cout << "Enter choice: ";
    cin >> choice;

    switch (choice)
    {
    case 0: 
        charucoboardCreate();
        break;

    case 1:
        detectCorners();
        break;

    case 2:
        calibrateCamera();
        break;

    case 3:
        poseEstimation();
        break;

    case 4:
        arOverlay();
        break;

    case 5: 
        featureDetection();
        break;

    case 6:
        arGL();
		break;

    default:
        cout << "Invalid choice" << endl;
    }
    return 0;
}