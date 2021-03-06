#include <stdio.h>
#include <opencv/cv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput);

int main( int argc, char** argv ) {
    char* imageName = argv[1];
    
    Mat image;
    image = imread(imageName, 1);
    
    if (argc != 2 || !image.data) {
        printf( " No image data \n " );
        return -1;
    }
    
    // CONVERT COLOUR, BLUR AND SAVE
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    
    Mat carBlurred;
    GaussianBlur(gray_image, 27, carBlurred);
    
    for (int k = 0; k < 5; k ++) {
        for( int i = 0; i < gray_image.rows; i++) {
            for( int j = 0; j < gray_image.cols; j++) {
                int val = gray_image.at<uchar>(i, j) + 2 * (gray_image.at<uchar>(i, j) - carBlurred.at<uchar>(i, j));
                if (val > 0 && val < 255) gray_image.at<uchar>(i, j) = (uchar)val;
            }
        }
    }
    
    
    imwrite("fixed.jpg", gray_image);
    imwrite("blur.jpg", carBlurred );
    
    return 0;
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput) {
    // intialise the output using the input
    blurredOutput.create(input.size(), input.type());
    
    // create the Gaussian kernel in 1D
    cv::Mat kX = cv::getGaussianKernel(size, -1);
    cv::Mat kY = cv::getGaussianKernel(size, -1);
    
    // make it 2D multiply one by the transpose of the other
    cv::Mat kernel = kX * kY.t();
    
    //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
    //TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!
    
    // we need to create a padded version of the input
    // or there will be border effects
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;
    
    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput,
        kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
        cv::BORDER_REPLICATE);
        
        // now we can do the convoltion
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                double sum = 0.0;
                for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
                    for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
                        // find the correct indices we are using
                        int imagex = i + m + kernelRadiusX;
                        int imagey = j + n + kernelRadiusY;
                        int kernelx = m + kernelRadiusX;
                        int kernely = n + kernelRadiusY;
                        
                        // get the values from the padded image and the kernel
                        int imageval = (int) paddedInput.at<uchar>(imagex, imagey);
                        double kernalval = kernel.at<double>(kernelx, kernely);
                        
                        // do the multiplication
                        sum += imageval * kernalval;
                    }
                }
                // set the output value as the sum of the convolution
                blurredOutput.at<uchar>(i, j) = (uchar) sum;
            }
        }
    }
    