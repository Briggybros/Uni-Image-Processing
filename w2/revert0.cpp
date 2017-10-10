#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

/*Image has had a hue rotation, I think. Trial and error to find that
blue had to  take the red values, green the blue values, and red the green
values.*/

int main() {

  // Read image from file
  Mat image = imread("mandrill0.jpg", 1);

  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     uchar pixelBlue = image.at<Vec3b>(y,x)[0];
     uchar pixelGreen = image.at<Vec3b>(y,x)[1];
     uchar pixelRed = image.at<Vec3b>(y,x)[2];

     image.at<Vec3b>(y,x)[0] = pixelRed;
     image.at<Vec3b>(y,x)[1] = pixelBlue;
     image.at<Vec3b>(y,x)[2] = pixelGreen;
     }
   }

   imwrite("newmandrill0.jpg", image);

  return 0;
}
