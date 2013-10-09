#include "visual_odometry.h"

int main(int argc, char** argv){
    VisualOdometry v;
    v.getRotation(cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE));
    cv::Mat rot = v.getRotation(cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE));
    std::cerr << rot << std::endl;
    return 0;
}
