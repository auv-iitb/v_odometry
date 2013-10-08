#include "only_rotation.cpp"

int main(int argc, char** argv){
    VisualOdometry v;
    v.getRotation(imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE));
    cv::Mat rot = v.getRotation(imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE));
    return 0;
}
