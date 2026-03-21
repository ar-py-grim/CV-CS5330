/*
Arpit Gandhi
March 2026

Different function definitions for ChArUco board creation, corner detection, camera calibration,
pose estimation, AR overlay, feature detection, and AR with OpenGL.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp> 
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include "utilities.h"

using namespace cv;
using namespace std;

const int squaresX = 10;
const int squaresY = 7;
const float squareSize = 1.0f;  // meters
const float markerSize = 0.6f;  // marker occupies 60% of the white square

// create and display ChArUco board, save on 's' key press, quit on 'q' key press
int charucoboardCreate()
{
    // DICT_6X6_250: 6x6 bit markers, 250 unique IDs
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    // create charuco board object
    aruco::CharucoBoard board(Size(squaresX, squaresY), squareSize, markerSize, dictionary);

    // generate board image
    Mat boardImage;
    board.generateImage(Size(1000, 700), boardImage, 10, 1);
    imshow("ChArUco Board", boardImage);

    while (true)
    {
        int key = waitKey(0) & 0xFF;
        // save file
        if (key == 's')
        {
            imwrite("charuco_board.png", boardImage);
        }
        else if (key == 'q')
        {
            destroyAllWindows();
            break;
        }
    }
    return 0;
}

// detect charuco corners in video stream, display count and first corner location, quit on 'q' key press
int detectCorners()
{
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard board(Size(squaresX, squaresY), squareSize, markerSize, dictionary);
    aruco::CharucoDetector detector(board);

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        vector<int>             markerIds;
        vector<vector<Point2f>> markerCorners;
        vector<int>             charucoIds;
        vector<Point2f>         charucoCorners;

        detector.detectBoard(frame, charucoCorners, charucoIds, markerCorners, markerIds);

        Mat display = frame.clone();

		// draw detected markers and corners
        if (!markerIds.empty())
        {
            aruco::drawDetectedMarkers(display, markerCorners, markerIds);
        }      

		// draw detected charuco corners and display count and first corner location
        if (!charucoIds.empty())
        {
            aruco::drawDetectedCornersCharuco(display, charucoCorners, charucoIds, Scalar(0, 255, 0));
            cout << "Corners found: " << charucoIds.size()
                << "  |  First corner [ID " << charucoIds[0] << "]: " << charucoCorners[0] << endl;
        }
        else
        {
            putText(display, "Board not detected", Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        putText(display, "Corners: " + to_string(charucoIds.size()), Point(20, display.rows-20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("ChArUco Detection", display);

        if ((waitKey(1) & 0xFF) == 'q')
        {
            destroyAllWindows();
            break;
        }
    }

    cap.release();
    return 0;
}

// camera calibration 's' to save frames with detected corners, 'c' to calibrate, 'q' to quit
int calibrateCamera()
{
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard board(Size(squaresX, squaresY), squareSize, markerSize, dictionary);
    aruco::CharucoDetector detector(board);

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    vector<vector<Point2f>> allCharucoCorners;
    vector<vector<int>>     allCharucoIds;
    Size imageSize;

    cout << "s = save frame  |  c = calibrate (min. 5 frames)  |  q = quit" << endl;

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        imageSize = frame.size();

        vector<int>             markerIds;
        vector<vector<Point2f>> markerCorners;
        vector<int>             charucoIds;
        vector<Point2f>         charucoCorners;

        detector.detectBoard(frame, charucoCorners, charucoIds, markerCorners, markerIds);

        Mat display = frame.clone();

        if (!markerIds.empty())
        {
            aruco::drawDetectedMarkers(display, markerCorners, markerIds);
        }  

        if (!charucoIds.empty())
        {
            aruco::drawDetectedCornersCharuco(display, charucoCorners, charucoIds, Scalar(0, 255, 0));
        }         

        putText(display, "Saved frames: " + to_string(allCharucoIds.size()), Point(20, display.rows-20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("Calibration", display);

        int key = waitKey(1) & 0xFF;

        if (key == 's')
        {
            if (!charucoIds.empty())
            {
                allCharucoCorners.push_back(charucoCorners);
                allCharucoIds.push_back(charucoIds);
                cout << "Frame saved. Total: " << allCharucoIds.size() << endl;
            }
            else
            {
                cout << "No board detected" << endl;
            }
        }
        else if (key == 'c')
        {
            if ((int)allCharucoIds.size() < 5)
            {
                cout << "Need at least 5 frames, current frame count: " << allCharucoIds.size() << endl;
                continue;
            }

			// initialize camera matrix
            Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
            cameraMatrix.at<double>(0,2) = imageSize.width/2.0;
            cameraMatrix.at<double>(1,2) = imageSize.height/2.0;

			// initialize camera distortion coefficients to zero
            Mat distCoeffs = Mat::zeros(4, 1, CV_64F);

            cout << "\nCamera Matrix Before:\n" << cameraMatrix << endl;
            cout << "\nDistortion Coeffs Before:\n" << distCoeffs << endl;

            vector<vector<Point3f>> allObjPoints;
            vector<vector<Point2f>> allImgPoints;

            for (int i=0; i<(int)allCharucoIds.size(); i++)
            {
                vector<Point3f> objPoints;
                vector<Point2f> imgPoints;
                board.matchImagePoints(allCharucoCorners[i], allCharucoIds[i], objPoints, imgPoints);
                if (objPoints.size()>=6)
                {
                    allObjPoints.push_back(objPoints);
                    allImgPoints.push_back(imgPoints);
                }
            }

            vector<Mat> rvecs, tvecs;
            double error = 0.0;

            try
            {
                error = calibrateCamera(allObjPoints, allImgPoints, imageSize,
                    cameraMatrix, distCoeffs, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO | CALIB_FIX_K3);
            }
            catch (const cv::Exception& e)
            {
                cerr << "Calibration failed: " << e.what() << endl;
                continue;
            }

            cout << "\nCamera Matrix After:\n" << cameraMatrix << endl;
            cout << "\nDistortion Coeffs After:\n" << distCoeffs << endl;
            cout << "\nReprojection Error: " << error << endl;

			// save calibration results to .yaml file
            FileStorage fs("config/calibration.yaml", FileStorage::WRITE);
            fs << "camera_matrix" << cameraMatrix;
            fs << "distortion_coeffs" << distCoeffs;
            fs << "reprojection_error" << error;
            fs.release();
        }

        else if (key == 'q')
        {
            destroyAllWindows();
            break;
        }
    }

    cap.release();
    return 0;
}

// estimate pose of charuco board, draw axes, quit on 'q' key press
int poseEstimation()
{
    Mat cameraMatrix, distCoeffs;
    FileStorage fs("config/calibration.yaml", FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Cannot open config/calibration.yaml." << endl;
        return -1;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coeffs"] >> distCoeffs;
    fs.release();

    cout << "Loaded Camera Matrix:\n" << cameraMatrix << endl;
    cout << "Loaded Distortion Coeffs:\n" << distCoeffs << endl;

    aruco::Dictionary      dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard    board(Size(squaresX, squaresY), squareSize, markerSize, dictionary);
    aruco::CharucoDetector detector(board);

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    cout << "q = quit" << endl;

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        vector<int>             markerIds;
        vector<vector<Point2f>> markerCorners;
        vector<int>             charucoIds;
        vector<Point2f>         charucoCorners;

        detector.detectBoard(frame, charucoCorners, charucoIds, markerCorners, markerIds);

        Mat display = frame.clone();

        if (!markerIds.empty())
        {
            aruco::drawDetectedMarkers(display, markerCorners, markerIds);
        }     

        if (charucoIds.size() >= 4)
        {
            vector<Point3f> objPoints;
            vector<Point2f> imgPoints;
            board.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);

			// only attempt pose estimation if we have at least 6 points to avoid instability
            if (objPoints.size()>=6)
            {
                Mat rvec, tvec;
                bool ok = false;
                try
                {
                    ok = solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
                }
                catch (const cv::Exception& e)
                {
                    cerr << "solvePnP failed: " << e.what() << endl;
                }
                if (ok)
                {
                    cout << "rvec: " << rvec.t() << "  |  tvec: " << tvec.t() << endl;
					// draw axes
                    vector<Point3f> axisPoints = {
                        {0, 0, 0}, {2, 0, 0},
                        {0, 0, 0}, {0, 2, 0},
                        {0, 0, 0}, {0, 0, 2}
                    };
                    vector<Point2f> projPoints;
                    projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, projPoints);

					// X-axis in red, Y-axis in green, Z-axis in blue
                    line(display, projPoints[0], projPoints[1], Scalar(0, 0, 255), 3);
                    line(display, projPoints[2], projPoints[3], Scalar(0, 255, 0), 3);
                    line(display, projPoints[4], projPoints[5], Scalar(255, 0, 0), 3);
                }
            }
        }
        else
        {
            putText(display, "Board not detected", Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }
        imshow("Pose Estimation", display);
        if ((waitKey(1) & 0xFF) == 'q')
        {
            destroyAllWindows();
            break;
        }
    }
    cap.release();
    return 0;
}

// fucntion to draw a cube on ChAruco board using the estimated pose, with rvec and tvec as input
static void drawCube(Mat& display, const Mat& rvec, const Mat& tvec, const Mat& cameraMatrix, const Mat& distCoeffs)
{
    vector<Point3f> cubePoints = {
        {0, 0, 0}, {2, 0, 0}, {2,-2, 0}, {0,-2, 0},
        {0, 0, 2}, {2, 0, 2}, {2,-2, 2}, {0,-2, 2}
    };

    vector<Point2f> projPoints;
    projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, projPoints);

    Scalar color(0, 200, 255);
	// thickness of cube edges
    int t = 2;
	// draw cube edges
    for (int i=0; i<4; i++)
    {
        line(display, projPoints[i], projPoints[(i+1)%4], color, t);
        line(display, projPoints[i+4], projPoints[(i+1)%4 + 4], color, t);
        line(display, projPoints[i], projPoints[i+4], color, t);
    }
}

// function to draw a torus (donut) on ChAruco board using the estimated pose, with rvec and tvec as input
static void drawDonut(Mat& display, const Mat& rvec, const Mat& tvec,
    const Mat& cameraMatrix, const Mat& distCoeffs)
{
    const int stepsU = 24;  // around the tube
    const int stepsV = 16;  // around the ring
    const float R = 1.5f;   // major radius
    const float r = 0.5f;   // minor radius
    const float cx = 1.5f, cy = -1.5f, cz = 1.5f; // center offset above board
	// lambda to convert (iu, iv) indices to 3D point on torus surface
    auto toPoint = [&](int iu, int iv) -> Point3f{
        float u = 2*CV_PI*iu/stepsU;
        float v = 2*CV_PI*iv/stepsV;
        return
        {
            // parametric torus formula
            cx + (R+ r*cos(v))*cos(u),
            cy + (R+ r*cos(v))*sin(u),
            cz + r*sin(v)
        };
    };

	// generate 3D points on the torus surface
    vector<Point3f> pts;
    for (int iu=0; iu<stepsU; iu++)
    {
        for (int iv=0; iv<stepsV; iv++)
        {
            pts.push_back(toPoint(iu, iv));
        }
    }
    vector<Point2f> p;
    projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs, p);

    Scalar color(0, 200, 255);
    int t = 1;
	// draw lines connecting each point to the next in u and v directions to form the torus mesh
    for (int iu=0; iu<stepsU; iu++)
    {
        for (int iv=0; iv<stepsV; iv++)
        {
            int curr = iu*stepsV + iv;
            int nextU = ((iu+1)%stepsU)*stepsV + iv;
            int nextV = iu*stepsV + (iv+1)%stepsV;

            line(display, p[curr], p[nextU], color, t);
            line(display, p[curr], p[nextV], color, t);
        }
    }
}

// function to draw a DNA double helix on ChAruco board using the estimated pose, with rvec and tvec as input
static void drawDNA(Mat& display, const Mat& rvec, const Mat& tvec,
    const Mat& cameraMatrix, const Mat& distCoeffs)
{
    const int steps = 60;       // points per strand
    const float height = 6.0f;  // total height of helix
    const float radius = 0.8f;  // helix radius
    const float turns = 2.5f;   // number of full turns
    const float cx = 1.5f, cy = -1.0f, cz = 0.0f;

    vector<Point3f> strand1, strand2;
	// generate points for two strands of the helix
    for (int i=0; i<=steps; i++)
    {
        float t = (float)i/steps;
        float angle = 2*CV_PI*turns*t;
        float z = cz+height*t;
        strand1.push_back({cx+radius*cosf(angle), cy+radius*sinf(angle), z});
        strand2.push_back({cx+radius*cosf(angle+(float)CV_PI), cy+radius*sinf(angle+(float)CV_PI), z});
    }

    vector<Point3f> allPts;
	// combine points from both strands for projection
    allPts.insert(allPts.end(), strand1.begin(), strand1.end());
    allPts.insert(allPts.end(), strand2.begin(), strand2.end());

    vector<Point2f> p;
    projectPoints(allPts, rvec, tvec, cameraMatrix, distCoeffs, p);

    int n = steps+1;

    // strand 1
    for (int i=0; i<steps; i++)
    {
        line(display, p[i], p[i+1], Scalar(255, 200, 0), 2);
    }   

    // strand 2
    for (int i=0; i<steps; i++)
    {
        line(display, p[n+i], p[n+i+1], Scalar(180, 0, 255), 2);
    } 

    // base pairs every 4 steps
    for (int i=0; i<=steps; i+= 4)
    {
        line(display, p[i], p[n+i], Scalar(0, 255, 255), 1);
    }
        
}

// add virtual 3D models (cube, donut, DNA) on ChAruco board using the estimated pose
int arOverlay()
{
    Mat cameraMatrix, distCoeffs;
    FileStorage fs("config/calibration.yaml", FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Cannot open config/calibration.yaml" << endl;
        return -1;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coeffs"] >> distCoeffs;
    fs.release();

    aruco::Dictionary      dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard    board(Size(squaresX, squaresY), squareSize, markerSize, dictionary);
    aruco::CharucoDetector detector(board);

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    cout << "c = Cube  |  d = Donut  |  n = DNA  |  q = quit" << endl;
    char mode = 'c';

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        vector<int>             markerIds;
        vector<vector<Point2f>> markerCorners;
        vector<int>             charucoIds;
        vector<Point2f>         charucoCorners;

        detector.detectBoard(frame, charucoCorners, charucoIds, markerCorners, markerIds);

        Mat display = frame.clone();

        if (!markerIds.empty())
            aruco::drawDetectedMarkers(display, markerCorners, markerIds);

        if (charucoIds.size() >= 4)
        {
            vector<Point3f> objPoints;
            vector<Point2f> imgPoints;
            board.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);

            if (objPoints.size()>=6)
            {
                Mat rvec, tvec;
                bool ok = false;
                try
                {
                    ok = solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
                }
                catch (const cv::Exception& e)
                {
                    cerr << "solvePnP failed: " << e.what() << endl;
                }
                if (ok)
					// switch between different 3D models to draw based on user input
                {
                    if (mode == 'c')
                    {
                        drawCube(display, rvec, tvec, cameraMatrix, distCoeffs);
                    }    
                    else if (mode == 'd')
                    {
                        drawDonut(display, rvec, tvec, cameraMatrix, distCoeffs);
                    }     
                    else if (mode == 'n')
                    {
                        drawDNA(display, rvec, tvec, cameraMatrix, distCoeffs);
                    }        
                }
            }
        }
        else
        {
            putText(display, "Board not detected", Point(40, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        string modeLabel = (mode == 'c') ? "Cube" : (mode == 'd') ? "Donut" : "DNA";
        putText(display, "Mode: " + modeLabel + "  (c/d/n)", Point(20, 40),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        imshow("AR Overlay", display);

        int key = waitKey(1) & 0xFF;

        if (key == 'c' || key == 'd' || key == 'n')
        {
            mode = (char)key;
        }
            
        else if (key == 'q')
        {
            destroyAllWindows();
            break;
        }
    }
    cap.release();
    return 0;
}

// Surf feature detection with adjustable hessian threshold, display count and first keypoint location
int featureDetection()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    int hessianThreshold = 400;

    cout << "SURF Feature Detection" << endl;
    cout << "=/-: adjust threshold  |  q: quit" << endl;

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
		// create SURF detector with current hessian threshold
        auto surf = xfeatures2d::SURF::create(hessianThreshold);
        vector<KeyPoint> keypoints;
        surf->detect(gray, keypoints);

        Mat display;
        cvtColor(gray, display, COLOR_GRAY2BGR);
		// draw detected keypoints and display count and first keypoint location
        drawKeypoints(display, keypoints, display, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        putText(display, "Threshold: " + to_string(hessianThreshold) + "  Features: " + to_string(keypoints.size()),
            Point(20, display.rows-20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("SURF Feature Detection", display);

        int key = waitKey(1) & 0xFF;
        if (key == 'q') 
        { 
            destroyAllWindows();
            break;
        }
		// adjust hessian threshold with = and - keys
        else if (key == '=' && hessianThreshold < 2000)
        {
            hessianThreshold+= 50;
        }   
        else if (key == '-' && hessianThreshold > 50)
        {
            hessianThreshold-= 50;
        }           
    }
    cap.release();
    return 0;
}
