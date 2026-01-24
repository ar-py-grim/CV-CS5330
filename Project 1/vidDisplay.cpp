/*
Arpit Gandhi
 January 2025

This file impelements a video display program that captures video from the camera
and applies various filters to the video stream in real-time. The user can switch
between different filter modes using keyboard inputs and save it to as video(.avi) 
when pressed 'r' key. User can also save cuurrent frame as image by pressing 's' key.

*/ 

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"
#include <chrono>

using namespace cv;
using namespace std;

// returns a double which gives time in seconds for Windows
double getTime()
{
    auto now = chrono::high_resolution_clock::now();   // Get current time
    auto duration = now.time_since_epoch();                 // Time since epoch
    return chrono::duration<double>(duration).count(); // Convert to seconds
}

// all filter modes
enum DisplayMode
{
    color,
    gray,
    alt_gray,
    sepia,
    cblur1,
    cblur2,
    xsobel,
    ysobel,
    color_grad,
    blur_quant,
    detect_faces,
    emboss,
    fog_effect,
    face_blur,
	face_highlight
};

// names of filter modes
const string MODE_NAMES[] ={"color", "gray", "alt_gray", "sepia", "cblur1", "cblur2",
    "xsobel", "ysobel", "color_grad", "blur_quant", "detect_faces", "emboss", 
    "fog_effect", "face_blur", "face_highlight"};

// number of times to run a filter for timing
const int Ntimes = 10;

// assigning keys to filter modes
DisplayMode getDisplayMode(char key)
{
    switch (key)
    {
    case 'c': return color;
    case 'g': return gray;
    case 'h': return alt_gray;
    case 'p': return sepia;
    case 'b': return cblur1;
    case 'u': return cblur2;
    case 'x': return xsobel;
    case 'y': return ysobel;
    case 'm': return color_grad;
    case 'l': return blur_quant;
    case 'f': return detect_faces;
    case 'e': return emboss;
    case 'z': return fog_effect;
    case 'k': return face_blur;
    case 'i': return face_highlight;
    default: return color;
    }
}

int main(int argc, char* argv[])
{
    VideoCapture* capdev;
    // open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return(-1);
    }
    // get some properties of the image
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH), (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
	int frame_width = (int)capdev->get(CAP_PROP_FRAME_WIDTH);
    int frame_height = (int)capdev->get(CAP_PROP_FRAME_HEIGHT);
	double fps = capdev->get(CAP_PROP_FPS);
	fps = fps>0 ? fps : 30.0;
    cout << "Expected size: " << refS.width << ", " << refS.height << endl;

    // create VideoWriter object
    VideoWriter output;
    bool isRecording = false;
    int rcnt = 1;

	// text parameters for recording indicator
    string text = "Recording";
    Point text_position(50, 50);
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    Scalar tcolor(255, 255, 255);
    int thickness = 2;
    int line_type = LINE_AA;

    namedWindow("Video");
	const String OUTPUT_DIR = "images/";

    // make a DANetwork object
    DA2Network depthNet("model_fp16.onnx");
    DisplayMode mode = color;
    Mat frame, opframe, sobel_x, sobel_y, gray_frame;

    for (;;)
    {
        // get a new frame from the camera, treat as a stream
        *capdev >> frame;
        if (frame.empty())
        {
            printf("empty frame");
            break;
        }

        char key = waitKey(10);

        if (key == 's')
        {
            // save current frame as image
            imwrite(OUTPUT_DIR + MODE_NAMES[mode] + ".png", opframe);
			cout << "Frame saved\n";
		}

        if (key == 'r')
        {
            if (!isRecording)
            {
                // start recording
                output.open(OUTPUT_DIR + "output" + to_string(rcnt) + ".avi",
                     VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height), true);
                if (!output.isOpened())
                {
                    cout << "Could not open the output video file for write\n";
                    return -1;
                }
                isRecording = true;
				rcnt++;
                cout << "Recording started\n";
            }
            else
            {
                // stop recording
                isRecording = false;
                cout << "Recording stopped\n";
                output.release();
			}
        }

        if (key == 'q')
        {
            break;
        }
        // valid key is pressed except q
        else if (key!= -1 && key!= 'q' && key!= 'r' && key!= 's')
        {
            DisplayMode newMode = getDisplayMode(key);
            // toggle to color mode
            if (mode == newMode && newMode!= color)
            {
                mode = color;
            }
            else
            {
                mode = newMode;
            }
        }

		// apply the selected mode 
        switch (mode)
        {
        case color:
            opframe = frame.clone();
            break;

        case gray:
            cvtColor(frame, opframe, COLOR_BGR2GRAY);
            break;

        case alt_gray:
            color_to_gray(frame, opframe);
            break;

        case sepia:
			sepia_filter(frame, opframe);
			break;

        case cblur1:
        {
            double startTime = getTime();
            // execute the file on the original image Ntimes
            for (int i = 0; i < Ntimes; i++)
            {
                blur5x5_1(frame, opframe);
            }
            // end the timing
            double endTime = getTime();
            // compute the time per image
            double difference = (endTime - startTime) / Ntimes;
            cout<<"Time per frame(1): "<<fixed<<setprecision(4)<<difference<<"s\n"<< endl;
        }
        break;

        case cblur2:
        {
            double startTime = getTime();
            // execute the file on the original image Ntimes
            for (int i = 0; i < Ntimes; i++)
            {
                blur5x5_2(frame, opframe);
            }
            // end the timing
            double endTime = getTime();
            // compute the time per image
            double difference = (endTime - startTime) / Ntimes;
            cout<<"Time per frame(2): "<<fixed<<setprecision(4)<<difference<<"s\n"<< endl;
        }
            break;

        case xsobel:
            sobelX3x3(frame, sobel_x);
            convertScaleAbs(sobel_x, opframe);
            break;

        case ysobel:
            sobelY3x3(frame, sobel_y);
            convertScaleAbs(sobel_y, opframe);
            break;

        case color_grad:
            sobelX3x3(frame, sobel_x);
            sobelY3x3(frame, sobel_y);
            magnitude(sobel_x, sobel_y, opframe);
            break;

        case blur_quant:
            blurQuantize(frame, opframe, 10);
            break;

        case detect_faces:
        {
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
            vector<Rect> faces;
            detectFaces(gray_frame, faces);
            frame.copyTo(opframe);
            drawBoxes(opframe, faces);
        }
        break;

        case emboss:
            embossing(frame, opframe);
            break;

        case fog_effect:
            fogEffect(frame, opframe, depthNet, 2.0);
            break;

        case face_blur:
        {
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
            vector<Rect> faces;
            detectFaces(gray_frame, faces);
            frame.copyTo(opframe);
            faceBlur(opframe, faces);
        }
        break;

        case face_highlight:
        {
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
            vector<Rect> faces;
            detectFaces(gray_frame, faces);
            frame.copyTo(opframe);
            faceHiglight(opframe, faces);
        }
        break;

        default:
            opframe = frame.clone();
            break;
        }

        if (isRecording)
        {
            putText(opframe, text, text_position, font_face, font_scale, tcolor, thickness, line_type);
        }
        // Display frame
        imshow("Video", opframe);

        // Write frame to output if recording
        if (isRecording && output.isOpened())
        {
            if (opframe.channels() == 1)
            {
                cvtColor(opframe, opframe, COLOR_GRAY2BGR);
                output.write(opframe);
            }
            else
            {
                output.write(opframe);
            }
        }
    }

	// clean up
    if (isRecording && output.isOpened())
    {
        output.release();
    }
    delete capdev;
    return(0);
}