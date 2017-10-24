#include <opencv/cv.h> //you may need to
#include <opencv/highgui.h> //adjust import locations
#include <opencv/cxcore.h> //depending on your machine setup

using namespace cv;

int main() {
    //create matrix for image
    Mat image;
    //loop through different thresholds
    for(int z = 1; z<25; z++){
        //load the mandrill image
        image = imread("mandrill.jpg", CV_LOAD_IMAGE_UNCHANGED);
        //goes through rows and columns of image
        for(int y = 0; y < image.rows; y++) { //go through all rows (or scanlines)
            for (int x = 0; x < image.cols; x++) { //go through all columns
                int val = (int)image.at<uchar>(y, x);
                if(val > z*10) {
                    image.at<uchar>(y, x) = 255;
                }
                else {
                    image.at<uchar>(y, x) = 0;
                }
            }
        }
        namedWindow("Display window", CV_WINDOW_AUTOSIZE);
        //visualise the loaded image in the window
        imshow("Display window", image);
        //wait for a key press until returning from the program
        waitKey(0);
        //free memory occupied by image
        image.release();
    }
    return 0;
}
