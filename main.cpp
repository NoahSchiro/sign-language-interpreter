#include <opencv2/opencv.hpp>

#include <iostream>

int main() {

	int hmin = 19, smin = 55, vmin = 107;
	int hmax = 54, smax = 221, vmax = 255;
	
	while(true) {

		//Image of a yellow bridge
		cv::Mat img = cv::imread("resources/BRIDGES.jpg");
	
		//Image with color detect
		cv::Mat imgHSV, imgOut;
	
		//Convert 
		cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
	
		//Bound the image to what we want
		cv::inRange(imgHSV, 
					cv::Scalar(hmin, smin, vmin),
					cv::Scalar(hmax, smax, vmax),
					imgOut);
	
		cv::imshow("Image", img);
		cv::imshow("ImageHSV", imgHSV);
		cv::imshow("ImageOut", imgOut);
		cv::waitKey(1);
	}

	return 0;
}