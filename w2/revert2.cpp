#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

/*Image was inverted, undone by simply taking each rgb value and
subtracting it from 255 to undo the invert.*/

int main() {

  // Read image from file
  Mat image = imread("mandrill2.jpg", 1);

  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     image.at<Vec3b>(y,x)[0] = 255 - image.at<Vec3b>(y,x)[0];
     image.at<Vec3b>(y,x)[1] = 255 - image.at<Vec3b>(y,x)[1];
     image.at<Vec3b>(y,x)[2] = 255 - image.at<Vec3b>(y,x)[2];
     }
   }

   imwrite("newmandrill2.jpg", image);

  return 0;
}
