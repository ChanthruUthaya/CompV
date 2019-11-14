// Numberplate.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <cmath>
#include <math.h>

using namespace std;
using namespace cv;

void sobel(char* name);
Mat sobeldx(cv::Mat& image);
Mat sobeldy(cv::Mat& image);
std::pair<cv::Mat, cv::Mat> sobelmag(cv::Mat& dx, cv::Mat& dy);
Mat sobeldir(cv::Mat& dx, cv::Mat& dy);
Mat thresholding(cv::Mat& grad, float thresh);
int*** malloc3dArray(int dim1, int dim2, int dim3);
std::pair<int, int> circleCenter(int radius, std::pair<int, int> pos, int grad);
void hough(cv::Mat& thresholdimage, cv::Mat& orienationimage);

int main(char* argc, char** argv) {

	char* name = argv[1];
	Mat image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat padded;
	cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
	Mat dx = sobeldx(padded);
	Mat dy = sobeldy(padded);
	std::pair<Mat, Mat> grads = sobelmag(dx, dy);
	Mat tmpgrad = grads.first;
	Mat outgrad = grads.second;
	Mat dir = sobeldir(dx, dy);
	imwrite("coins1/dx.jpg", dx);
	imwrite("coins1/dy.jpg", dy);
	imwrite("coins1/maggrad.jpg", outgrad);
	imwrite("coins1/dirgrad.jpg", dir);
	Mat thresh = thresholding(outgrad, 80.0);
	imwrite("coins1/thresh.jpg", thresh);
	hough(thresh,dir);
}


Mat thresholding(cv::Mat& grad, float thresh) {
	Mat thresholded(grad.rows, grad.cols, DataType<int>::type, Scalar(0));
	for (int y = 0; y < grad.rows; y++) {
		for (int x = 0; x < grad.cols; x++) {
			if (grad.at<float>(y, x) < thresh) {
				thresholded.at<int>(y, x) = 0;
				}
			else{
				thresholded.at<int>(y, x) = 255;
			}
		}
	}
	return thresholded;
}

void hough(cv::Mat& thresholdimage, cv::Mat& orienationimage) {

	int minr = 20;
	int maxr = 50;
	int offset = minr;
	int dim1 = thresholdimage.cols;
	int dim2 = thresholdimage.rows;
	int dim3 = maxr - minr + 1;

	int*** hougharr = malloc3dArray(dim1, dim2, dim3);

	for (int y = 0; y < thresholdimage.rows; y++) {
		for (int x = 0; x < thresholdimage.cols; x++) {
			if (thresholdimage.at<int>(y, x) == 255) {
				for (int r = minr; r < maxr + 1; r++) {
					float grad = orienationimage.at<float>(y, x);
					std::pair<int,int> p = circleCenter(r, std::make_pair(x,y), grad);
					int x0 = p.first;
					int y0 = p.second;
					if (0 <= x0 <= thresholdimage.cols - 1 && 0 <= y0 <= thresholdimage.rows - 1) {
						hougharr[x0][y0][r-minr] += 1;
						//std::cout << "vote, ";
					}
				}
			}
		}
	}
}



std::pair<int, int> circleCenter(int radius, std::pair<int,int> pos, int grad) {
	int x = pos.first;
	int y = pos.second;
	int x0 = x + 1;  //floor(radius * cos(grad));
	int y0 = y + 1; //floor(radius * sin(grad));


	return std::make_pair(x0, y0);
}

void sobel(char* name) {
	Mat image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat padded;
	cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
	Mat dx = sobeldx(padded);
	Mat dy = sobeldy(padded);
	//Mat grad = sobelmag(dx, dy);
	Mat dir = sobeldir(dx, dy);
	imwrite("coins1/dx.jpg", dx);
	imwrite("coins1/dy.jpg", dy);
	//imwrite("coins1/maggrad.jpg", grad);
	imwrite("coins1/dirgrad.jpg", dir);

}

Mat sobeldx(cv::Mat& image) {
	Mat temp(image.rows - 2, image.cols - 2, DataType<int>::type, Scalar(0));
	for (int y = 1; y < temp.rows + 1; y++) {
		for (int x = 1; x < temp.cols + 1; x++) {
			int result = 0;
			result += -(image.at<uchar>(y - 1, x - 1));
			result += -2*(image.at<uchar>(y, x - 1));
			result += -(image.at<uchar>(y + 1, x - 1));
			result += (image.at<uchar>(y - 1, x + 1));
			result += 2*(image.at<uchar>(y, x + 1));
			result += (image.at<uchar>(y + 1, x +1));
			temp.at<int>(y-1, x-1) = result;
		}
	}
	return temp;
}

Mat sobeldy(cv::Mat& image) {
	Mat temp(image.rows-2, image.cols-2, DataType<int>::type, Scalar(0));
	for (int y = 1; y < temp.rows+1; y++) {
		for (int x = 1; x < temp.cols+1; x++) {
			int result = 0;
			result += -(image.at<uchar>(y-1, x - 1));
			result += -2*(image.at<uchar>(y-1, x));
			result += -(image.at<uchar>(y-1, x + 1));
			result += (image.at<uchar>(y+1, x + 1));
			result += 2*(image.at<uchar>(y+1, x));
			result += (image.at<uchar>(y+1, x - 1));
			temp.at<int>(y-1, x-1) = result;
		}
	}

	return temp;
}


std::pair<cv::Mat, cv::Mat> sobelmag(cv::Mat& dx, cv::Mat& dy) {
	Mat temp(dx.rows, dx.cols, DataType<float>::type, Scalar(0));
	Mat output(dx.rows, dx.cols, DataType<float>::type, Scalar(0));
	for (int y = 0; y < dx.rows; y++) {
		for (int x = 0; x < dx.cols; x++) {
			int tbs = dy.at<int>(y, x)*dy.at<int>(y, x) + dx.at<int>(y, x)*dx.at<int>(y, x);
			temp.at<float>(y, x) =  sqrt(tbs);
		}
	}
	cv::normalize(temp, output, 0, 255, NORM_MINMAX);
	return std::make_pair(temp,output);
}

Mat sobeldir(cv::Mat& dx, cv::Mat& dy) {
	Mat temp(dx.rows, dx.cols, DataType<int>::type, Scalar(0));
	Mat output(dx.rows, dx.cols, DataType<float>::type, Scalar(0));
	for (int y = 0; y < dx.rows; y++) {
		for (int x = 0; x < dx.cols; x++) {
			if (dx.at<int>(y, x) == 0) {
				if (dy.at<int>(y, x) < 0) {
					temp.at<float>(y, x) = 3.141 / 2;
				}
				else {
					temp.at<float>(y, x) = - 3.141 / 2;
				}
			}
			else {
				temp.at<float>(y, x) = atan(dy.at<int>(y, x) / dx.at<int>(y, x));
			}
			//std::cout << temp.at<float>(y, x);
		}
	}
	//cv::normalize(temp, output, 0, 255, NORM_MINMAX);
	return temp;
}

int*** malloc3dArray(int dim1, int dim2, int dim3)
{
	int i, j, k;
	int*** array = (int***)malloc(dim1 * sizeof(int**));

	for (i = 0; i < dim1; i++) {
		array[i] = (int**)malloc(dim2 * sizeof(int*));
		for (j = 0; j < dim2; j++) {
			array[i][j] = (int*)malloc(dim3 * sizeof(int));
		}

	}

	for (i = 0; i < dim1; ++i)
		for (j = 0; j < dim2; ++j)
			for (k = 0; k < dim3; ++k)
				array[i][j][k] = 0;

	return array;
}