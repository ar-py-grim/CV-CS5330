/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Generate a histogram, display a visualization of it, and saves it.
*/


#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
using namespace cv;

// Main function looks for a command line argument
int main(int argc, char *argv[])
{

  Mat src;
  Mat dst;  // viewing image
  Mat hist; // histogram data
  Mat chromImg; // chromaticity image

  const char filename[256] = "images/HW_img.tif";
  float max;
  const int histsize = 256;

  // read the file
  src = imread(filename);  
  if(src.data == NULL)
  {
    printf("error: unable to read filename %s\n", filename);
    return(-1);
  }
   
  // create an image to hold the chromaticity values
  chromImg.create(src.size(), CV_8UC3);
  // initialize the histogram (use floats so we can make probabilities)
  hist = Mat::zeros(Size(histsize, histsize), CV_32FC1);

  // keep track of largest bucket for visualization purposes
  max = 0.0;

  // loop over all pixels
  for(int i=0; i<src.rows; i++)
  {
    Vec3b *ptr = src.ptr<Vec3b>(i); // pointer to row i
	Vec3b* cptr = chromImg.ptr<Vec3b>(i); // pointer to row i
    for(int j=0; j<src.cols; j++)
    {
      // get the RGB values
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];

      // compute the r,g chromaticity
      float divisor = R+G+B;
      divisor = divisor > 0.0f ? divisor : 1.0f; // check for all zeros
      float r = R/divisor;
      float g = G/divisor;
      float b = 1.0f-(r+g);

      float intensity = 250.0f;
	  cptr[j][0] = (uchar)(b*intensity);
	  cptr[j][1] = (uchar)(g*intensity);
	  cptr[j][2] = (uchar)(r*intensity);

      // compute indexes, r, g are in [0, 1]
      int rindex = (int)(r*(histsize-1)+0.5);
      int gindex = (int)(g*(histsize-1)+0.5);

      // increment the histogram
      hist.at<float>(rindex, gindex)++;

      // keep track of the size of the largest bucket (just so we know what it is)
      float newvalue = hist.at<float>(rindex, gindex);
      max = newvalue>max ? newvalue : max;
    }
  }

  // print information
  printf("The largest bucket has %d pixels in it\n", (int)max);
  
  // histogram is complete
  // normalize the histogram by the number of pixels
  hist/= (src.rows*src.cols); // divides all elements of a cv::Mat by a scalar

  // the chromaticity histogram is ready for saving as a feature vector now

  // make a visualization of a histogram
  // color each bin by its chromaticity and weight
  dst.create(hist.size(), CV_8UC3);
  for(int i=0;i<hist.rows;i++)
  {
    Vec3b *ptr = dst.ptr<Vec3b>(i);
    float *hptr = hist.ptr<float>(i);
    for(int j=0;j<hist.cols;j++)
    {
      if(i+j>hist.rows)
      {
        ptr[j] = Vec3b(200, 120, 60); // default color
        continue;
      }
      
      float rcolor = (float)i/histsize;
      float gcolor = (float)j/histsize;
      float bcolor = 1-(rcolor+gcolor);

      ptr[j][0] = hptr[j]>0 ? hptr[j]*128 + 128*bcolor : 0;
      ptr[j][1] = hptr[j]>0 ? hptr[j]*128 + 128*gcolor : 0;
      ptr[j][2] = hptr[j]>0 ? hptr[j]*128 + 128*rcolor : 0;
    }
  }

  // original iamge
  resize(src, src, Size(src.cols/4, src.rows/8));
  imshow("Original",src);
  // whole histogram is in the range [0,1]
  imshow("rg Histogram", dst);
  // chromaticity image
  resize(chromImg, chromImg, Size(chromImg.cols/4, chromImg.rows/8));
  imshow("Chromaticity Image", chromImg);

  while (true)
  {
      // save the histogram to a file
      int key = waitKey(0) & 0xFF;
      if (key == 's')
      {
          imwrite("images/rg_histogram.png", dst);
		  imwrite("images/chromaticity_image.png", chromImg);
          printf("File Saved\n");
      }
      else if (key == 'q')
      {
          destroyAllWindows();
          break;
      }
  }
  return(0);
}
