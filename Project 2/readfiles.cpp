/*
  Bruce A. Maxwell
  S21
  
  Sample code to identify image files in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "dirent.h"
#include <opencv2/opencv.hpp>

using namespace cv;
/*
  Given a directory, scans through the directory for image files.
  Prints out the full path name for each file. This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[])
{
  char dirname[256], buffer[256] = {};
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  printf("Processing directory %s\n", dirname);

  // check for sufficient arguments
  if (argc<2)
  {
      printf("usage: %s <directory path>\n", argv[0]);
      exit(-1);
  }
  // get the directory path
  strcpy_s(dirname, argv[1]);
  printf("Processing directory %s\n", dirname);

  // open the directory
  dirp = opendir(dirname);
  if(dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while((dp = readdir(dirp))!= NULL)
  {

    // check if the file is an image
    if(strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) 
    {
      printf("processing image file: %s\n", dp->d_name);
      // build the overall filename
      strcpy_s(buffer, dirname);
      strcat_s(buffer, "/");
      strcat_s(buffer, dp->d_name);
      printf("full path name: %s\n", buffer);
    }
  }
  printf("Terminating\n");
  // showing last image in directory
  imshow("Image", imread(buffer));
  waitKey(0);
  destroyAllWindows();

  return(0);
}


