/*
Arpit Gandhi
February 2025

Implementation of feature extraction and distance metrics

*/

#include "features.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

vector<float> extract_center(Mat& image)
// Extracts a 7x7 pixel region from the center of the image
{
    vector<float> features;
    // Pre-allocate for efficiency
    features.reserve(7*7*3);  
    int center_x = image.cols/2;
    int center_y = image.rows/2;
    // Extract 7x7 region centered at image center
    for (int i=-3; i<=3; i++)
    {
		// row pointer
        Vec3b* ptr = image.ptr<Vec3b>(center_y+i);
        for (int j=-3; j<=3; j++)
        {
            features.push_back(ptr[center_x+j][0]); // B
            features.push_back(ptr[center_x+j][1]); // G
            features.push_back(ptr[center_x+j][2]); // R
        }
    }
    return features;
}

float ssd(vector<float>& f1, vector<float>& f2)
// Computes ssd between two feature vectors of the same size
{
    if (f1.size()!= f2.size())
    {
        printf("Feature vectors have different sizes\n");
        return -1.0;
    }
    float sum = 0.0;
    for (int i=0; i<f1.size(); i++)
    {
        sum+= (f1[i]-f2[i])*(f1[i]-f2[i]);
    }
    return sum;
}

vector<float> rg_histogram(Mat& image, int histsize)
{
    vector<float> histogram(histsize*histsize, 0.0f);
    int total_pixels = 0;
	// loop over all pixels
    for (int i=0; i<image.rows; i++)
    {
        Vec3b* ptr = image.ptr<Vec3b>(i);
        for (int j=0; j<image.cols; j++)
        {
            float blue = (float)ptr[j][0];
            float green = (float)ptr[j][1];
            float red = (float)ptr[j][2];
            float divisor = red+green+blue;
            // check for all zeros
            divisor = divisor>0.0f ? divisor : 1.0f; 
            float r = red/divisor;
            float g = green/divisor;
            // compute bin indexes
            int rindex = (int)(r*(histsize-1)+0.5);
            int gindex = (int)(g*(histsize-1)+0.5);
            // 2D index to 1D index
            int bin_index = (rindex*histsize)+gindex;
            // increment histogram count
            histogram[bin_index]+= 1.0f;
            total_pixels++;
        }
    }

    // normalize histogram
    for (int i=0; i<histogram.size(); i++)
    {
        histogram[i]/= total_pixels;
    }
    return histogram;
}

vector<float> rgb_histogram(Mat& image, int histsize)
{
    vector<float> histogram(histsize*histsize*histsize, 0.0f);
    int total_pixels = 0;
    // loop through all pixels
    for (int i=0; i<image.rows; i++)
    {
        Vec3b* ptr = image.ptr<Vec3b>(i);
        for (int j=0; j<image.cols; j++)
        {
            // normalize to [0, 1]
            float b = ptr[j][0]/255.0f;
            float g = ptr[j][1]/255.0f;
            float r = ptr[j][2]/255.0f;

            // compute bin indexes
            int bindex = (int)(b*(histsize-1)+0.5);
            int gindex = (int)(g*(histsize-1)+0.5);
            int rindex = (int)(r*(histsize-1)+0.5);

            // convert 3D index to 1D index
            int bin_index = (rindex*histsize*histsize) + (gindex*histsize) + bindex;
            histogram[bin_index]+= 1.0f;
            total_pixels++;
        }
    }

	// normalize histogram
    for (int i=0; i<histogram.size(); i++)
    {
        histogram[i]/= total_pixels;
    }
    return histogram;
}

float hist_inter(std::vector<float>& h1, std::vector<float>& h2)
{
    if (h1.size()!= h2.size())
    {
        printf("Histograms have different sizes\n");
        return -1.0;
    }
    float intersection = 0.0;
    // sum of minimum values in each bin
    for (int i=0; i<h1.size(); i++)
    {
        intersection+= min(h1[i], h2[i]);
    }
    // convert intersection to distance
    float distance = 1.0f-intersection;
    return distance;
}

int sobelX3x3(Mat& src, Mat& dst)
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
    for (int i=1; i<r-1; i++)
    {
        for (int j=1; j<c-1; j++)
        {
            long long bsum = 0, gsum = 0, rsum = 0;
            // [0] = Blue, [1] = Green, [2] = Red
            for (int ki=-1; ki<=1; ki++)
            {
                Vec3b* sptr = src.ptr<Vec3b>(i+ki);
                int* kptr = sobel_matrix.ptr<int>(ki+1);
                for (int kj=-1; kj<=1; kj++)
                {
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

int sobelY3x3(Mat& src, Mat& dst)
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
    for (int i=1; i<r-1; i++)
    {
        for (int j=1; j<c-1; j++)
        {
            long long bsum = 0, gsum = 0, rsum = 0;
            // [0] = Blue, [1] = Green, [2] = Red
            for (int ki=-1; ki<=1; ki++)
            {
                Vec3b* sptr = src.ptr<Vec3b>(i+ki);
                int* kptr = sobel_matrix.ptr<int>(ki+1);
                for (int kj=-1; kj<=1; kj++)
                {
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

int magnitude(Mat& sx, Mat& sy, Mat& dst)
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

vector<float> texture_histogram(Mat& image, int histsize)
{
    Mat sobelX, sobelY, img;
    sobelX3x3(image, sobelX);
    sobelY3x3(image, sobelY);
    magnitude(sobelX, sobelY, img);
    Mat gray_img = Mat::zeros(img.rows, img.cols, CV_32F);
    for (int i=0; i<img.rows; i++)
    {
        Vec3b* mptr = img.ptr<Vec3b>(i);
        float* gptr = gray_img.ptr<float>(i);
        for (int j=0; j<img.cols; j++)
        {
            gptr[j] = (mptr[j][0]+mptr[j][1]+mptr[j][2])/3.0f;
        }
    }

    vector<float> histogram(histsize, 0.0f);
    int total_pixels = 0;

    for (int i=0; i<gray_img.rows; i++)
    {
        float* gptr = gray_img.ptr<float>(i);
        for (int j=0; j<gray_img.cols; j++)
        {
            float mag = gptr[j]/255.0f;
            int bin = (int)(mag*(histsize-1) + 0.5);
            histogram[bin]+= 1.0f;
            total_pixels++;
        }
    }

    // normalize
    for (int i=0; i<histogram.size(); i++)
    {
        histogram[i]/= total_pixels;
    }
    return histogram;
}

vector<float> gabor_features(Mat& image, int gsize, int num_theta, int num_scales)
{
    vector<float> features;
    Mat img, fimg, kernel;
    // convert to grayscale
    cvtColor(image, img, COLOR_BGR2GRAY);
    // normalize to [0, 1]
    img.convertTo(img, CV_32F, 1.0/255.0);

    double sigma = 4.0;
    double lambda_base = 8.0;
    double gamma = 0.5;
    double psi = 0;

    // apply Gabor filters at different orientations and scales
    for (int s=0; s<num_scales; s++)
    {
        double lambda = lambda_base*pow(2,s);
        for (int o=0; o<num_theta; o++)
        {
            double theta = o * (CV_PI/num_theta);
            kernel = getGaborKernel(Size(gsize, gsize), sigma, theta, lambda, gamma, psi);
            filter2D(img, fimg, CV_32F, kernel);

            // compute mean and std. deviation
            Scalar mean, stddev;
            meanStdDev(fimg, mean, stddev);

            features.push_back(mean[0]);
            features.push_back(stddev[0]);
        }
    }
    return features;
}
