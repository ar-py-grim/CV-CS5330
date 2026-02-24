/*
Arpit Gandhi

This file implements various image filter functions.

*/ 

#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"
using namespace cv;
using namespace std;

int color_to_gray(Mat &src, Mat &dst)
// convert color frame to grayscale
{
	int r = src.rows;
	int c = src.cols;
	src.copyTo(dst);
	for (int i=0; i<r; i++)
	{
		Vec3b *dptr = dst.ptr<Vec3b>(i);
		for (int j=0; j<c; j++)
		{
			int val = (int)(255-dptr[j][2]);
			// same value for all 3 channels
			dptr[j][0] = val;
			dptr[j][1] = val;
			dptr[j][2] = val;
		}
	}
	return 0;
}

int sepia_filter(Mat &src, Mat &dst, int levels)
// apply sepia tone and vignetting effect
{
	int r = src.rows;
	int c = src.cols;
	src.copyTo(dst);
	Mat schannels[3], dchannels[3], gx, gy, gkernel;
	split(src, schannels);
	// sepia tone conversion
	dchannels[0] = 0.131*schannels[0] + 0.534*schannels[1] + 0.272*schannels[2];
	dchannels[1] = 0.168*schannels[0] + 0.686*schannels[1] + 0.349*schannels[2];
	dchannels[2] = 0.189*schannels[0] + 0.769*schannels[1] + 0.393*schannels[2];
	for (int i=0; i<3; i++)
	{
		// Max. value 255
		min(dchannels[i], 255.0, dchannels[i]); 
		// convert to 8 bit image
		dchannels[i].convertTo(dchannels[i], CV_8U); 
	}
	merge(dchannels, 3, dst);

	// vignetting effect

	int rows = (r%2==0) ? r+1 : r; // odd number of rows
	int cols = (c%2==0) ? c+1 : c; // odd number of cols
	gx = getGaussianKernel(cols,rows/(double)levels, CV_64F);
	gy = getGaussianKernel(rows,cols/(double)levels, CV_64F);
	gkernel = gy*gx.t();
	// crop the kernel to image size
	gkernel = gkernel(Rect(0, 0, c, r));
	normalize(gkernel, gkernel, 0, 1, NORM_MINMAX);
	split(dst, dchannels);
	for (int i=0; i<3; i++)
	{
		dchannels[i].convertTo(dchannels[i], CV_64F);
		dchannels[i] = dchannels[i].mul(gkernel);
		// convert to 8 bit image
		dchannels[i].convertTo(dchannels[i], CV_8U);
	}
	merge(dchannels, 3, dst);
	return 0;
}

int blur5x5_1(Mat &src, Mat &dst)
// apply 5x5 Gaussian blur using .at<> method
{
	int r = src.rows;
	int c = src.cols;
	int blur_kernel[5][5] = {
		{1, 2, 4, 2, 1},
		{2, 4, 8, 4, 2},
		{4, 8, 16, 8, 4},
		{2, 4, 8, 4, 2},
		{1, 2, 4, 2, 1}
	};
	Mat blur_matrix(5, 5, CV_32S, blur_kernel);
	flip(blur_matrix, blur_matrix, -1);
	src.copyTo(dst);
	for (int i=2; i<r-2; i++)
	{
		for (int j=2; j<c-2; j++)
		{
			long long bsum=0, gsum=0, rsum=0;
			for (int ki=-2; ki<=2; ki++)
			{
				for (int kj=-2; kj<=2; kj++)
				{
					long long weight = blur_matrix.at<int>(ki+2, kj+2);
					bsum+= src.at<Vec3b>(i+ki, j+kj)[0]*weight;
					gsum+= src.at<Vec3b>(i+ki, j+kj)[1]*weight;
					rsum+= src.at<Vec3b>(i+ki, j+kj)[2]*weight;
				}
			}
			// normalizing channel values
			int blue = min(255, (int)(bsum/100));
			int green = min(255, (int)(gsum/100));
			int red = min(255, (int)(rsum/100));
			dst.at<Vec3b>(i, j) = Vec3b(blue, green, red);
		}
	}
	return 0;
}

int blur5x5_2(Mat &src, Mat &dst)
// apply 5x5 Gaussian blur using ptr<> method and separable kernel
{
	int kernel[5] = {1, 2, 4, 2, 1};
	int r = src.rows;
	int c = src.cols;
	Mat temp;
	src.copyTo(temp);
	for (int i=0; i<r; i++)
	{
		// src row pointer
		Vec3b* sptr = src.ptr<Vec3b>(i);  
		// temp row pointer
		Vec3b* tptr = temp.ptr<Vec3b>(i); 
		for (int j=2; j<c-2; j++)
		{
			int bsum=0, gsum=0, rsum=0;
			// apply 1×5 horizontal kernel
			for (int kj=-2; kj<=2; kj++)
			{
				int weight = kernel[kj+2];
				bsum+= sptr[j+kj][0]*weight;
				gsum+= sptr[j+kj][1]*weight;
				rsum+= sptr[j+kj][2]*weight;
			}
			tptr[j] = Vec3b(min(255, bsum/10), min(255, gsum/10), min(255, rsum/10));
		}
	}
	temp.copyTo(dst);
	for (int i=2; i<r-2; i++)
	{
		// dst row pointer
		Vec3b* dptr = dst.ptr<Vec3b>(i);  
		for (int j=0; j<c; j++)
		{
			int bsum=0, gsum=0, rsum=0;
			// apply 5×1 vertical kernel
			for (int ki=-2; ki<=2; ki++)
			{
				int weight = kernel[ki+2];
				Vec3b* tptr = temp.ptr<Vec3b>(i+ki);  
				bsum+= tptr[j][0]*weight;
				gsum+= tptr[j][1]*weight;
				rsum+= tptr[j][2]*weight;
			}
			dptr[j] = Vec3b(min(255, bsum/10), min(255, gsum/10), min(255, rsum/10));
		}
	}
	return 0;
}

int sobelX3x3(Mat &src, Mat &dst)
// apply Sobel filter in X direction
{
	int sobelx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	int r = src.rows;
	int c = src.cols;
	dst = Mat::zeros(r, c, CV_16SC3);
	Mat sobel_matrix(3, 3, CV_32S, sobelx);
	//flip(sobel_matrix, sobel_matrix, -1);
	for (int i=1; i<r-1; i++)
	{
		for (int j=1; j<c-1; j++)
		{
			long long bsum=0, gsum=0, rsum=0;
			// [0] = Blue, [1] = Green, [2] = Red
			for (int ki=-1; ki<=1; ki++)
			{
				Vec3b *sptr = src.ptr<Vec3b>(i+ki);
				int *kptr = sobel_matrix.ptr<int>(ki+1);
				for (int kj=-1; kj<=1; kj++)
				{	
					//long long weight = sobel_matrix.at<int>(ki+1,kj+1);
					long long weight = (long long)kptr[kj+1];
					bsum+= sptr[j+kj][0]*weight;
					gsum+= sptr[j+kj][1]*weight;
					rsum+= sptr[j+kj][2]*weight;
				}
			}
			dst.at<Vec3s>(i,j) = Vec3s((short)bsum, (short)gsum, (short)rsum);
		}
	}
	return 0;
}

int sobelY3x3(Mat &src, Mat &dst)
// apply Sobel filter in Y direction
{	
	int sobely[3][3] = {
		{1,   2,  1},
		{0,   0,  0},
		{-1, -2, -1}
	};
	int r = src.rows;
	int c = src.cols;
	dst = Mat::zeros(r, c, CV_16SC3);
	Mat sobel_matrix(3, 3, CV_32S, sobely);
	//flip(sobel_matrix, sobel_matrix, -1);
	for (int i=1; i<r-1; i++)
	{
		for (int j=1; j<c-1; j++)
		{
			long long bsum=0, gsum=0, rsum=0;
			// [0] = Blue, [1] = Green, [2] = Red
			for (int ki=-1; ki<=1; ki++)
			{
				Vec3b *sptr = src.ptr<Vec3b>(i+ki);
				int *kptr = sobel_matrix.ptr<int>(ki+1);
				for (int kj=-1; kj<=1; kj++)
				{
					long long weight = (long long)kptr[kj+1];
					//bsum+= src.at<Vec3b>(i+ki,j+kj)[0]*weight;
					bsum+= sptr[j+kj][0]*weight;
					gsum+= sptr[j+kj][1]*weight;
					rsum+= sptr[j+kj][2]*weight;
				}
			}
			dst.at<Vec3s>(i, j) = Vec3s((short)bsum, (short)gsum, (short)rsum);
		}
	}
	return 0;
}

int magnitude(Mat &sx, Mat &sy, Mat &dst)
// compute magnitude of gradients
{
	int r = sx.rows;
	int c = sx.cols;
	dst = Mat::zeros(r, c, CV_8UC3);
	for (int i=0; i<r; i++)
	{
		// pointer to row i in sobelx
		Vec3s* sxptr = sx.ptr<Vec3s>(i); 
		// pointer to row i in sobely
		Vec3s* syptr = sy.ptr<Vec3s>(i);    
		// pointer to row i in dst
		Vec3b* dptr = dst.ptr<Vec3b>(i); 
		for (int j=0; j<c; j++)
		{
			for (int ch=0; ch<3; ch++)
			{
				double val = (double)(sqrt((sxptr[j][ch]*sxptr[j][ch]) + (syptr[j][ch]*syptr[j][ch])));
				dptr[j][ch] = (uchar)(min(255.0, val));
			}
		}
	}
	return 0;
}

int blurQuantize(Mat &src, Mat &dst, int levels)
// apply blur and quantization effect
{
	Mat blurred;
	blur5x5_2(src, blurred);
	int r = blurred.rows;
	int c = blurred.cols;
	dst = Mat::zeros(r, c, blurred.type());
	// bucket size
	int b = 255/levels;
	for(int i=0; i<r; i++)
	{
		Vec3b* blurrptr = blurred.ptr<Vec3b>(i);
		Vec3b* dptr = dst.ptr<Vec3b>(i);        
		for (int j=0; j<c; j++)
		{
			for (int ch=0; ch<3; ch++)
			{
				int x = blurrptr[j][ch];
				// quantize pixel value
				int xt = x/b;
				// value after quantization
				int xf = xt*b;
				dptr[j][ch] = (uchar)(min(255, xf));
			}
		}
	}
	return 0;
}

int embossing(Mat &src, Mat &dst)
// apply embossing effect with orientation at 45 degrees
{
	Mat sx, sy;
	sobelX3x3(src, sx);
	sobelY3x3(src, sy);
	int r = src.rows;
	int c = src.cols;
	dst = Mat::zeros(r, c, src.type());
	// light source direction 45 degrees
	float light_x = 1.0/sqrt(2.0);
	float light_y = 1.0/sqrt(2.0);
	for(int i=0; i<r; i++)
	{
		Vec3s* sxptr = sx.ptr<Vec3s>(i);
		Vec3s* syptr = sy.ptr<Vec3s>(i);
		Vec3b* dptr = dst.ptr<Vec3b>(i);        
		// dot product with light source direction
		for (int j=0; j<c; j++)
		{
			for (int ch=0; ch<3; ch++)
			{
				float dot = (sxptr[j][ch]*light_x) + (syptr[j][ch]*light_y);
				int val = (int)(dot+128);
				dptr[j][ch] = (uchar)(min(255, max(0, val)));
			}
		}
	}
	return 0;
}

int faceBlur(Mat &frame, vector<Rect> &faces, int ksize, float scale, double sigx, double sigy)
// blur frame aroud detected faces
{
	Mat blurred;
	// blur entire frame
	GaussianBlur(frame, blurred, Size(ksize,ksize), sigx, sigy);
	for (int i=0; i<faces.size(); i++)
	{
		Rect face = faces[i];
		if (scale!= 1.0)
		{
			face.x = (int)(face.x*scale);
			face.y = (int)(face.y*scale);
			face.width = (int)(face.width*scale);
			face.height = (int)(face.height*scale);
		}
		face.x = max(0, face.x);
		face.y = max(0, face.y);
		face.width = min(frame.cols-face.x, face.width);
		face.height = min(frame.rows-face.y, face.height);
		// copy face region to blurred frame
		if (face.width>0 && face.height>0)
		{
			frame(face).copyTo(blurred(face));
		}
	}
	blurred.copyTo(frame);
	return 0;
}

int fogEffect(Mat &src, Mat &dst, DA2Network &depthNet, int fog_intensity, float density)
// apply exponential fog effect using depth and fog density
{
	Mat small_src, depthMap;
	resize(src, small_src, Size(), 0.5, 0.5);
	depthNet.set_input(small_src, 1.0);
	depthNet.run_network(depthMap, src.size());
	int r = src.rows;
	int c = src.cols;
	dst = Mat::zeros(r, c, src.type());
	Vec3b fog_matrix(fog_intensity, fog_intensity, fog_intensity);
	depthMap.convertTo(depthMap, CV_32FC1);
	// normalize depth value to [0, 1]
	normalize(depthMap, depthMap, 0.0, 1.0, NORM_MINMAX);
	for (int i=0; i<r; i++)
	{
		Vec3b* sptr = src.ptr<Vec3b>(i);
		float* depthptr = depthMap.ptr<float>(i);
		Vec3b* dptr = dst.ptr<Vec3b>(i);
		for (int j=0; j<c; j++)
		{
			// inverting depth value
			float d = 1-depthptr[j];
			for (int ch=0; ch<3; ch++)
			{
				// fog increaces with depth
				float fog_amount = 1.0f-exp(-d*density);
				fog_amount = min(1.0f, max(0.0f, fog_amount));
				dptr[j][ch] = (uchar)(sptr[j][ch]*(1.0f-fog_amount) + fog_matrix[ch]*fog_amount);
			}
			
		}
	}
	return 0;
}

int faceHiglight(Mat &frame, vector<Rect> &faces, float scale)
// highlight detected faces while rest of frame is grayscale
{
	Mat grey;
	// grey scale entire frame
	color_to_gray(frame, grey);
	for (int i=0; i<faces.size(); i++)
	{
		Rect face = faces[i];
		if (scale!= 1.0)
		{
			face.x = (int)(face.x*scale);
			face.y = (int)(face.y*scale);
			face.width = (int)(face.width*scale);
			face.height = (int)(face.height*scale);
		}
		face.x = max(0, face.x);
		face.y = max(0, face.y);
		face.width = min(frame.cols-face.x, face.width);
		face.height = min(frame.rows-face.y, face.height);
		// copy face region to grey frame
		if (face.width>0 && face.height>0)
		{
			frame(face).copyTo(grey(face));
		}
	}
	grey.copyTo(frame);
	return 0;
}

int cannyEdge(Mat &src, Mat &dst, int threshold)
// Canny edge detection
{
	Mat blurred, sx, sy;
	GaussianBlur(src, blurred, Size(5,5), 0.0);

	sobelX3x3(blurred, sx);
	sobelY3x3(blurred, sy);

	int r = sx.rows;
	int c = sx.cols;
	dst = Mat::zeros(r, c, CV_8UC3);

	for (int i=1; i<r-1; i++)
	{
		Vec3s* sxptr = sx.ptr<Vec3s>(i);
		Vec3s* syptr = sy.ptr<Vec3s>(i);
		Vec3b* dptr = dst.ptr<Vec3b>(i);

		for (int j=1; j<c-1; j++)
		{
			// Average across channels
			float gx = (sxptr[j][0] + sxptr[j][1] + sxptr[j][2])/3.0f;
			float gy = (syptr[j][0] + syptr[j][1] + syptr[j][2])/3.0f;

			float mag = sqrt(gx*gx + gy*gy);

			// thresholding
			uchar val = (mag>threshold) ? 255 : 0;
			dptr[j] = Vec3b(val, val, val);
		}
	}

	return 0;

}
