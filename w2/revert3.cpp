#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

/*Image was clearly just applied a BGR->HSV conversion, doing the opposite
mostly saves the image, although the conversion causes some irregularities*/

int main() {
  // Read image from file
  Mat image = imread("mandrill3.jpg", 1);

  // CONVERT AND SAVE THE IMAGE
  cvtColor( image, image, CV_HSV2BGR );
  imwrite( "newmandrill3.jpg", image );

  return 0;
}
