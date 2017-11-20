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
void detectAndDisplay( Mat frame );
float jaccardIndex(Rect rect1, Rect rect2);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

    // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

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
