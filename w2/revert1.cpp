#include <stdio.h>
#include <opencv/cv.hpp>

using namespace cv;

/*RED SI ISSUE.*/

int main() {
    
    // Read image from file
    Mat image = imread("mandrill1.jpg", 1);
    int z = 30;
    
    for(int y=0; y<image.rows; y++) {
        for(int x=0; x<image.cols; x++) {
            int newY = (((image.rows - y) - z) + image.rows) % image.rows;
            int newX = (((image.cols - x) - z) + image.cols) % image.cols;
            uchar pixelOld = image.at<Vec3b>(newY, newX)[2];
            image.at<Vec3b>((image.rows - y), (image.cols - x))[2] = pixelOld;
        }
    }
    
    imwrite("newmandrill1.jpg", image);
    
    return 0;
}
