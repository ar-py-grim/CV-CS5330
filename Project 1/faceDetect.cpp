/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them

  The path to the Haar cascade file is define in faceDetect.h
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"
#include "filter.h"
using namespace cv;
using namespace std;

/*
  Arguments:
  cv::Mat grey  - a greyscale source image in which to detect faces
  std::vector<cv::Rect> &faces - a standard vector of cv::Rect rectangles indicating where faces were found
     if the length of the vector is zero, no faces were found
 */

int detectFaces(Mat &grey, vector<Rect> &faces)
{
  // a static variable to hold a half-size image
  static Mat half;
  
  // a static variable to hold the classifier
  static CascadeClassifier face_cascade;

  // the path to the haar cascade file
  static String face_cascade_file(FACE_CASCADE_FILE);

  if(face_cascade.empty())
  {
    if(!face_cascade.load(face_cascade_file))
    {
      printf("Unable to load face cascade file\n");
      printf("Terminating\n");
      exit(-1);
    }
  }

  // clear the vector of faces
  faces.clear();
  
  // cut the image size in half to reduce processing time
  resize(grey, half, Size(grey.cols/2, grey.rows/2));

  // equalize the image
  equalizeHist(half, half);

  // apply the Haar cascade detector
  face_cascade.detectMultiScale(half, faces);

  // adjust the rectangle sizes back to the full size image
  for(int i=0; i<faces.size(); i++)
  {
    faces[i].x *= 2;
    faces[i].y *= 2;
    faces[i].width *= 2;
    faces[i].height *= 2;
  }
  return(0);
}

/* Draws rectangles into frame given a vector of rectangles
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minSize - ignore rectangles with a width small than this argument
   float scale - scale the rectangle values by this factor (in case frame is different than the source image)
 */
int drawBoxes(Mat &frame, vector<Rect> &faces, int minWidth, float scale)
{
  // The color to draw, you can change it here (B, G, R)
  Scalar wcolor(170, 120, 110);

  for(int i=0; i<faces.size(); i++)
  {
    if(faces[i].width > minWidth)
    {
      Rect face(faces[i]);
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      rectangle(frame, face, wcolor, 3);
    }
  }
  return(0);
}