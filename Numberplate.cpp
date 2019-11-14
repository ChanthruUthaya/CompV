// Numberplate.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <cmath>

using namespace std;
using namespace cv;

void GaussianBlur2d(
	cv::Mat& input,
	int size,
	cv::Mat& blurredOutput);

void GaussianBlurConv(
	cv::Mat& input,
	int size,
	cv::Mat& blurredOutput);

Mat filter2d(char* name);
int convolution(int argc, char** argv);
void task2(char* name);
uchar median_val(cv::Mat& kernel, int width);
Mat kernelMatrix(cv::Mat& padded, int x_cor, int y_cor, int window_size);
void task1(char* name);

int main(int argc, char** argv)
{
	char* name = argv[1];
	//task1(name);
	task2(name);

}

void task2(char* name) {
	int window_size = 3;
	Mat noisey_image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	int border = floor(window_size / 2);
	Mat padded;
	cv::copyMakeBorder(noisey_image, padded, border, border, border, border, cv::BORDER_REPLICATE);
	Mat update(noisey_image.rows, noisey_image.cols, DataType<uchar>::type, Scalar(0));
	for (int y= border; y < noisey_image.rows + border; y++) {
		for (int x = border; x < noisey_image.cols + border; x++) {
			Mat kernel = kernelMatrix(padded, x, y, window_size);
			uchar val = median_val(kernel, window_size);
			update.at<uchar>(y-border, x-border) = val;
		}
	}
	imwrite("salt_pepper.jpg", update);
}

uchar median_val(cv::Mat& kernel, int width) {
	kernel = kernel.reshape(0, 1);
	std::vector<uchar> vect;  
	kernel.copyTo(vect);
	std::sort(vect.begin(), vect.end());
	return vect[((width*width)-1)/2];
}

Mat kernelMatrix(cv::Mat& padded, int x_cor, int y_cor, int window_size) {
	int kernelwidth = (floor(window_size / 2));
	int kernelheight = kernelwidth;
	Mat kernel(window_size, window_size, DataType<uchar>::type, Scalar(0));
	for (int y = -kernelheight; y < kernelheight + 1; y++) {
		for (int x = -kernelwidth; x < kernelwidth + 1 ; x++) {
			kernel.at<uchar>(y + kernelheight, x + kernelwidth) = padded.at<uchar>(y_cor + y, x_cor + x);
		}
	}
	

	return kernel;
}

void task1(char* name) {

	Mat gray_image;
	cvtColor(imread(name, 1), gray_image, CV_BGR2GRAY);
	Mat blur_image = filter2d(name);
	Mat img_tmp(blur_image.rows, blur_image.cols, DataType<int>::type, Scalar(0));
	for (int i = 0; i < 20; i++) {
		for (int y = 0; y < blur_image.rows; y++) {
			for (int x = 0; x < blur_image.cols; x++) {
				img_tmp.at<int>(y, x) = (2 * (gray_image.at<uchar>(y, x)) - (blur_image.at<uchar>(y, x)));
			}
		}
	}

	imwrite("sharp.jpg", img_tmp);
}

int convolution(int argc, char** argv)
{

	// LOADING THE IMAGE
	char* imageName = argv[1];

	Mat image;
	image = imread(imageName, 1);

	if (argc != 2 || !image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	// CONVERT COLOUR, BLUR AND SAVE
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	Mat carBlurred;
	GaussianBlurConv(gray_image, 23, carBlurred);

	imwrite("blur.jpg", carBlurred);

	return 0;
}

Mat filter2d(char* name)
{

	// LOADING THE IMAGE

	Mat image;
	image = imread(name, 1);

	// CONVERT COLOUR, BLUR AND SAVE
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	Mat carBlurred;
	GaussianBlur2d(gray_image, 5, carBlurred);

	imwrite("blur.jpg", carBlurred);

	return carBlurred;
}



void GaussianBlurConv(cv::Mat& input, int size, cv::Mat& blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);

	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernalval = kernel.at<double>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar)sum;
		}
	}
}

void GaussianBlur2d(cv::Mat& input, int size, cv::Mat& blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	// SET KERNEL VALUES
	for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
		for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
			kernel.at<double>(m + kernelRadiusX, n + kernelRadiusY) = (double)1.0 / (size * size);

	}

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);

	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernalval = kernel.at<double>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar)sum;
		}
	}
}
