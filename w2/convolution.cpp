#include <stdio.h>
#include <iostream>
#include <cmath>
#include <opencv/cv.hpp>

using namespace cv;
using namespace std;

/*RED SI ISSUE.*/

int main() {
    
    // Read image from file
    Mat image = imread("mandrill.jpg", 1);
    
    int z = 5;
    
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    
    for(int y=z; y<gray_image.rows - z; y++) {
        for(int x=z; x<gray_image.cols - z; x++) {
            
            int sum = 0;
            
            for(int u=-z; u<1+z; u++){
                for(int v=-z; v<1+z; v++){
                    int current = (int)gray_image.at<uchar>(y+u, x+v);
                    sum = sum + current;
                }
            }
            
            
            sum = sum/pow((2*z)+1, 2);
            
            gray_image.at<uchar>(y, x) = (uchar)sum;
        }
    }
    
    imwrite("convol.jpg", gray_image);
    
    return 0;
}
