#ifndef VISUAL_ODOM_HH
#define VISUAL_ODOM_HH
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/tracking.hpp"

class VisualOdometry{
    public:
        VisualOdometry(){};
        cv::Mat getRotation(cv::Mat img);
    private:
        cv::Mat img1;
        cv::Mat img2;

};
#endif
