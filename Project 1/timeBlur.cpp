/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/

#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include <chrono>
#include "opencv2/opencv.hpp" // general opencv header file
#include "filter.h"
using namespace cv;

// returns a double which gives time in seconds for Windows
double getTime()
{
    auto now = std::chrono::high_resolution_clock::now();   // Get current time
	auto duration = now.time_since_epoch();                 // Time since epoch
    return std::chrono::duration<double>(duration).count(); // Convert to seconds
}  

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char *argv[])  // main function, execution starts here
{  
  Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
  Mat dst; // cv::Mat to hold the output of the process
  char filename[256]; // a string for the filename

  // usage: checking if the user provided a filename
  if(argc < 2)
  {
    printf("Usage %s <image filename>\n", argv[0]);
    exit(-1);
  }
  strcpy_s(filename, argv[1]); // copying 2nd command line argument to filename variable

  // read the image
  src = imread(filename); // allocating the image data
  // test if the read was successful
  if(src.data == NULL)   // src.data is the reference to the image data
  {  
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }
  printf("src image type= %d\n", src.type());
  const int Ntimes = 10;
	
  ////////////////set up the timing for version 1/////////////////
  double startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0; i<Ntimes; i++)
  {
    blur5x5_1(src, dst);
  }
  printf("dst image type= %d\n", dst.type());
  imshow("input image", src);
  imshow("blur1", dst);

  // end the timing
  double endTime = getTime();

  // compute the time per image
  double difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("Time per image (1): %.4lf seconds\n", difference);

  /////////////////set up the timing for version 2/////////////////
  startTime = getTime();
  // execute the file on the original image a couple of times
  for(int i=0; i<Ntimes; i++)
  {
    blur5x5_2(src, dst);
  }
  imshow("blur2", dst);

  // end the timing
  endTime = getTime();

  // compute the time per image
  difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("Time per image (2): %.4lf seconds\n", difference);
  
  if ((waitKey(0) & 0xFF) == 'q')
  {
      destroyAllWindows();
  }

  // terminate the program
  printf("Terminating\n");

  return(0);
}
