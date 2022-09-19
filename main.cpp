#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>

//Contour finding function
void getContours(cv::Mat imgIn, cv::Mat imgOut) {
}


int main() {
	
	cv::VideoCapture cap(0);

	cv::Mat img, imgProcessed;

	cv::CascadeClassifier faceClass;
	faceClass.load("resources/haarcascade_frontalface_default.xml");

	//Pointer check
	if(faceClass.empty()) {
		std::cout << "Error! File path invalid!";
	}

	//Store all faces in frame
	std::vector<cv::Rect> faces;

	while(true) {

		//Read in image from webcam
		cap.read(img);

		//Resize to be larger
		cv::resize(img, imgProcessed, cv::Size(), 1.7, 1.7, cv::INTER_LINEAR);

		//Detect faces
		faceClass.detectMultiScale(imgProcessed, faces, 1.1, 10);

		//For each face
		for(int i = 0; i < faces.size(); i++) {

			//Draw the bounding box for the face
			cv::rectangle(imgProcessed, faces[i].tl(), faces[i].br(), cv::Scalar(255, 255, 255), 3);
			cv::putText(imgProcessed, "Ugly", faces[i].tl(), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(255,255,255), 3);

		}

		cv::imshow("Webcam", imgProcessed);

		//Wait a milisecond
		cv::waitKey(1);
	}



	return 0;
}