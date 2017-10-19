// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <iostream>
#include <math.h>

#define PI 3.14159265

using namespace cv;
using namespace std;

//defining one function to do all convolutions
//dir is 0 or 1, and computes the change in x or y respectively
//val determines the middle values, ie, 1 or 2.
void convolute( cv::Mat &input, int dir, int val, cv::Mat &output){

  // intialise the output using the input
  output.create(input.size(), input.type());

  //must start and end 1 in to prevent OOB errors.
  for( int i = 0; i < input.rows; i++ ){
    for( int j = 0; j < input.cols; j++ ){
      double sum = 0.0;
      //begin filter loop
      int z = 0;

      for( int x = -1; x<2; x++ ){
        if(i+x < 0 || i+x >= input.rows) sum += 0;
        else {
          for( int y = -1; y<2; y++ ){
            if(j+y < 0 || j+y >= input.cols) sum += 0;
            if( dir == 0) z = y;
            if( dir == 1) z = x;
            if( x == 0 || y == 0) z *= val;
            else {
              sum += (z * input.at<uchar>(i+x, j+y));
            }
          }
        }
      }
      if(sum < 0) sum *= -1;
      output.at<uchar>(i, j) = (uchar)(sum);
    }
  }
}

void sobel ( cv::Mat &input){

  Mat xChangeImage;
  convolute(input, 0, 2, xChangeImage);

  Mat yChangeImage;
  convolute(input, 1, 2, yChangeImage);

  Mat magImage;
  Mat dirImage;
  magImage.create(input.size(), input.type());
  dirImage.create(input.size(), input.type());
  for(int i = 0; i < input.rows; i++){
    for(int j = 0; j < input.cols; j++){
      int x = xChangeImage.at<uchar>(i, j);
      int y = yChangeImage.at<uchar>(i, j);
      magImage.at<uchar>(i, j) = (uchar)sqrt((x*x) + (y*y));
      if( x == 0 ) dirImage.at<uchar>(i, j) = 90;
      else{
        dirImage.at<uchar>(i, j) = atan (y/x) * 180/PI;
      }
    }
  }



  namedWindow("Display window", CV_WINDOW_AUTOSIZE);
  imshow("Display window", xChangeImage);
  waitKey(0);

  imshow("Display window", yChangeImage);
  waitKey(0);

  imshow("Display window", magImage);
  waitKey(0);

  imshow("Display window", dirImage);
  waitKey(0);

  xChangeImage.release();
  yChangeImage.release();
  magImage.release();
  dirImage.release();
}


int main ( int argc, char** argv){
  // LOADING THE IMAGE
  char* imageName = argv[1];


  Mat image;
  image = imread( imageName, 1 );

  if( argc != 2 || !image.data )
  {
    printf( " No image data \n " );
    return -1;
  }

  // CONVERT COLOUR
  Mat gray_image;
  cvtColor( image, gray_image, CV_BGR2GRAY );

  sobel( gray_image );

  return 0;

}
