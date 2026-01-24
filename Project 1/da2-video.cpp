/*
  Bruce A. Maxwell
  January 2025

  An example of using the DA2Network class with OpenCV for a video stream

*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
using namespace cv;

// opens a video stream and runs it through the depth anything network
// displays both the original video stream and the depth stream
int main(int argc, char *argv[])
{
  VideoCapture *capdev;
  Mat src; 
  Mat dst;
  Mat dst_vis;
  const float reduction = 0.5;

  // make a DANetwork object
  DA2Network da_net("model_fp16.onnx");

  // open the video device
  capdev = new VideoCapture(0);
  if(!capdev->isOpened())
  {
    printf("Unable to open video device\n");
    return(-1);
  }

  Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH), (int)capdev->get(CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  float scale_factor = 256.0 / (refS.height*reduction);
  printf("Using scale factor %.2f\n", scale_factor);

  /*
  namedWindow("video", 1);
  namedWindow("depth", 2);
  */

  for(;;)
  {
    // capture the next frame
    *capdev >> src;
    if( src.empty())
    {
      printf("frame is empty\n");
      break;
    }
    // for speed purposes, reduce the size of the input frame by half
    resize(src, src, cv::Size(), reduction, reduction);

    // set the network input
    da_net.set_input(src, scale_factor);

    // run the network
    da_net.run_network(dst, src.size());

    // apply a color map to the depth output to get a good visualization
    applyColorMap(dst, dst_vis, COLORMAP_INFERNO);

    // if you want to modify the src image based on the depth image, do that here
    /*
    for(int i=0;i<src.rows;i++)
    {
      for(int j=0;j<src.cols;j++)
      {
	    if(dst.at<unsigned char>(i, j) < 128)
        {
	      src.at<Vec3b>(i,j) = Vec3b(128, 100, 140);
	    }
      }
    }
    */

    // display the images
    imshow("video", src);
    imshow("depth", dst_vis);

    // terminate if the user types 'q'
    char key = waitKey(10);
    if(key == 'q')
    {
      break;
    }
  }

  printf("Terminating\n");

  return(0);
}

