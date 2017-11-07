#include <opencv/cv.hpp>

using namespace cv;

int main() {
    //create matrix for image
    Mat image;
    //load the mandrill image
    image = imread("mandrillRGB.jpg", CV_LOAD_IMAGE_UNCHANGED);
    //goes through rows and columns of image
    for(int y = 0; y < image.rows; y++) { //go through all rows (or scanlines)
        for (int x = 0; x < image.cols; x++) { //go through all columns
            int val = (int)image.at<Vec3b>(y, x)[2]; //set to blue, green or red
            if (val > 220) {
                image.at<Vec3b>(y, x)[0] = 255;
                image.at<Vec3b>(y, x)[1] = 255;
                image.at<Vec3b>(y, x)[2] = 255;
            } else {
                image.at<Vec3b>(y, x)[0] = 0;
                image.at<Vec3b>(y, x)[1] = 0;
                image.at<Vec3b>(y, x)[2] = 0;
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
    return 0;
}
