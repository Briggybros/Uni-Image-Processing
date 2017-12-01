/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <opencv/cv.hpp>
#include <iostream>
#include <stdio.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/** Ground truths for all images */
const vector<vector<Rect>> groundTruths = {
	{Rect(443,14,154,180)},
	{Rect(196,132,195,191)},
	{Rect(101,95,91,93)},
	{Rect(323,148,67,72)},
	{Rect(185,94,213,203)},
	{Rect(433,141,110,111)},
	{Rect(210,115,63,66)},
	{Rect(254,170,150,144)},
	{Rect(842,218,117,119), Rect(67,252,60,89)},
	{Rect(203,48,231,232)},
	{Rect(92,104,95,109), Rect(585,127,56,86), Rect(916,149,37,65)},
	{Rect(174,105,59,56)},
	{Rect(156,77,60,137)},
	{Rect(272,120,131,131)},
	{Rect(120,101,125,127), Rect(989,95,122,125)},
	{Rect(154,56,129,138)},
};

/** Function Headers */
void detectAndDisplay( Mat frame, bool improved, Mat hough );
float f1(vector<Rect> detections);
float jaccardIndex(Rect rect1, Rect rect2);
void edgeDetection(Mat grad, Mat dir, Mat input, int scale, int delta, int thresh);
void houghLines(Mat image, Mat grad);
void bumpCols(Mat image, int thresh);
void houghCircles(Mat out, Mat mag, Mat dir, int min_r, int max_r);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
int largestRadius = 115;
int CIRCLE_THRESHOLD = 150; // default threshold value
int MAX_HOUGH = 0;
int HOUGH_THRESHOLD = 150;

/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){
		printf("--(!)Error loading\n"); return -1;
	};

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, 0, frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	//Hough Implementation

	//Image preparation
	Mat splitH[3];
	Mat houghLab = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	//Split image into separate layers
	split(houghLab, splitH);
	Mat tests;
	tests = splitH[2].clone();
	// cvtColor(houghLab, tests, CV_BGR2GRAY);
	imwrite( "b.jpg", splitH[2]); //R channel makes the dartboards very clear
	GaussianBlur(tests, tests, Size(3,3), 0, 0, BORDER_DEFAULT);
	Mat gradient, direction;
	gradient.create(tests.size(), tests.type());
	direction.create(tests.size(), tests.type());

	//Edge detection
	edgeDetection(gradient, direction, tests, 1, 0, 140);

	imwrite( "grad_mag.jpg", gradient);
	imwrite( "grad_dir.jpg", direction);

	//Hough circles
	Mat circles;
	circles.create(gradient.size(), gradient.type());
	houghCircles(circles, gradient, direction, 17, largestRadius);
	imwrite( "hough_circles.jpg", circles);

	//Hough lines
	Mat lines;
	lines.create(gradient.size(), gradient.type());
	lines = Scalar::all(0); //was getting odd noise on empty image.
	houghLines(lines, gradient);
	imwrite( "hough_lines.jpg", lines);

	//Detect and display improved detector
	detectAndDisplay(houghLab, 1, circles);
	imwrite("detected2.jpg", houghLab);

	return 0;
}

//Make red pixels black, and green pixels white.
void bumpCols(Mat input, int thresh){
	for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++) {
					int green = input.at<Vec3b>(i, j)[1];
					int red = input.at<Vec3b>(i, j)[2];
					if(red > thresh){
						input.at<Vec3b>(i, j)[0] = 255;
						input.at<Vec3b>(i, j)[1] = 255;
						input.at<Vec3b>(i, j)[2] = 255;
					}
					else if(thresh < green){
						input.at<Vec3b>(i, j)[0] = 0;
						input.at<Vec3b>(i, j)[1] = 0;
						input.at<Vec3b>(i, j)[2] = 0;
					}
			}
	}
}

int*** create3DArray(int x, int y, int z){
	int ***array = new int**[x];
	for (int i = 0; i<x; i++){
		array[i] = new int*[y];
		for(int j = 0; j<y; j++){
			array[i][j] = new int[z];
		}
	}
	return array;
}

void houghLines(Mat out, Mat mag){
	vector<Vec2f> lines;
	HoughLines(mag, lines, 1, CV_PI/180, HOUGH_THRESHOLD, 0, 0);

	for(int i = 0; i < lines.size(); i++){
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
	  pt1.y = cvRound(y0 + 1000*(a));
	  pt2.x = cvRound(x0 - 1000*(-b));
	  pt2.y = cvRound(y0 - 1000*(a));
	  line( out, pt1, pt2, Scalar(255), 1, CV_AA);
	}
}

void houghCircles(Mat out, Mat mag, Mat dir, int min_r, int max_r){
	int height = mag.rows, width = mag.cols, depth = largestRadius - min_r;
	int*** circles = create3DArray(width, height, depth);
	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0; x < mag.cols; x++) {
			if(mag.at<uchar>(y, x) == 255){
				float t = dir.at<uchar>(y, x);
				for (int r = min_r; r < max_r; r++){
					//get circle centers
					int x0 = x - r * cos(t*CV_PI/180);
					int y0 = y - r * sin(t*CV_PI/180);
					if ((x0 >= 0 && y0 >= 0) && (x0 < width && y0 < height)){
						circles[x0][y0][(r - min_r)] += 1;
					}
					x0 = x + r * cos(t*CV_PI/180);
					y0 = y + r * sin(t*CV_PI/180);
					if ((x0 >= 0 && y0 >= 0) && (x0 < width && y0 < height)){
						circles[x0][y0][(r - min_r)] += 1;
					}
				}
			}
		}
	}

	//Flatten 3d array into 2d.
	for (int y = 0; y < mag.rows; y++){
		for (int x = 0; x < mag.cols; x++){
			int sum = 0;
			for (int r = 0; r < (largestRadius-min_r); r++) {
				sum += circles[x][y][r];
			}
			if(sum > 255){
				sum = 255;
			}
			out.at<uchar>(y, x) = sum;
			if(sum > MAX_HOUGH && x > 3){
				MAX_HOUGH = sum;
			}
		}
	}
	if(MAX_HOUGH < CIRCLE_THRESHOLD || MAX_HOUGH > (CIRCLE_THRESHOLD + 50)){
		CIRCLE_THRESHOLD = MAX_HOUGH*0.8;
	}
}

void edgeDetection( Mat gradient, Mat direction, Mat input, int scale, int delta, int thresh ) {

	Mat d_x, d_y, abs_x, abs_y;

	//X Gradient
	Sobel( input, d_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//Y Gradient
	Sobel( input, d_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	//Convert results to CV_8U
	convertScaleAbs( d_x, abs_x);
	convertScaleAbs( d_y, abs_y);

	//Create gradient using absolute values instead of squaring
	addWeighted(abs_x, 0.5, abs_y, 0.5, 0, gradient);

	//Threshold and calculate direction
	for (int i = 0; i < gradient.rows; i++) {
			for (int j = 0; j < gradient.cols; j++) {
					//Thresholding
					int z = gradient.at<uchar>(i, j);
					if (z > thresh) gradient.at<uchar>(i, j) = 255;
					else gradient.at<uchar>(i, j) = 0;

					//Direction stuff
          int x = d_x.at<short>(i, j);
          int y = d_y.at<short>(i, j);
					if (y <= 20) direction.at<uchar>(i, j) = 0;
					else if (x == 0) direction.at<uchar>(i, j) = 90;
					else {
							int val = atan2(y, x)*180/CV_PI;
							direction.at<uchar>(i, j) = (uchar)val;
					}
			}
	}
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, bool improved, Mat hough)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	if( improved ){
		std::vector<Rect> newFaces;
		for (int i = 0; i<faces.size(); i++){
			int width = faces[i].width / 2;
			int height = faces[i].height / 2;
			int x0 = faces[i].x + (width/2);
			int y0 = faces[i].y + (height/2);
			bool center = false;
			for (int x = x0; x<(width + x0); x++){
				for (int y = y0; y<(height + y0); y++){
					if(hough.at<uchar>(y, x) > CIRCLE_THRESHOLD) center = true;
				}
			}
			if(center) newFaces.push_back(faces[i]);
		}
		faces = newFaces;
	}

    // 3. Print number of Faces found
	if(improved){
		std::cout << "Detected faces (Improved): " << faces.size() << std::endl;
	}
	else {
		std::cout << "Detected faces: " << faces.size() << std::endl;
	}

    // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

float f1(vector<Rect> detections, vector<Rect> groundTruths) {
	const float jaccardThreshold = 0.9;

	int correctDetections = 0;
	int boardsFound = 0;

	for (int i = 0; i < (int)groundTruths.size(); i++) {
		bool detected = false;
		for (int j = 0; j < (int)detections.size(); i++) {
			if (jaccardIndex(groundTruths[i], detections[j]) > jaccardThreshold) {
				detected = true;
				correctDetections++;
			}
		}

		if (detected) {
			boardsFound++;
		}
	}

	float precision = (float)boardsFound/(float)(detections.size() - correctDetections);
	float recall = (float)boardsFound/(float)groundTruths.size();

	return 2 * ((precision * recall)/(precision + recall));
}

float jaccardIndex(Rect rect1, Rect rect2) {
	float xOverlap = std::max(0, std::min(rect1.x + rect1.width, rect2.x + rect2.width) - std::max(rect1.x, rect2.x));
    float yOverlap = std::max(0, std::min(rect1.y + rect1.height, rect2.y + rect2.height) - std::max(rect1.y,rect2.y));

	if (xOverlap == 0 || yOverlap == 0) return 0;

	float intersection = xOverlap * yOverlap;

	float rect1Size = rect1.width * rect1.height;
	float rect2Size = rect2.width * rect2.height;

	float uni = (rect1Size + rect2Size) - intersection; // union is keyword in C++

	return intersection / uni;
}
