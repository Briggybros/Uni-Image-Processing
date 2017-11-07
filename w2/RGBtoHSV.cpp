/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - RGBtoHSV.cpp
//
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.hpp>

using namespace cv;

int main(int argc, char** argv) {
    
    // LOADING THE IMAGE
    char* imageName = argv[1];
    
    Mat image;
    image = imread(imageName, 1);
    
    if(argc != 2 || !image.data) {
        printf(" No image data \n ");
        return -1;
    }
    
    // CONVERT AND SAVE THE IMAGE
    cvtColor(image, image, CV_BGR2HSV);
    imwrite("hsv.jpg", image);
    
    return 0;
}
