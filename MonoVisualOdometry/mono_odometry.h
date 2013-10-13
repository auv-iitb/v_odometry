#ifndef MONO_ODOMETRY_H
#define MONO_ODOMETRY_H

#include <cmath>
#include <time.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


class MonoVisualOdometry {

public:

  // camera parameters (all are mandatory / need to be supplied)
 /*
  struct calibration {  
    float fx;  // f/dx (in pixels)
    float fy;  // f/dy (in pixels)    
    float uo; // principal point (u-coordinate)
    float vo; // principal point (v-coordinate)
    calibration () { //for resolution: 320x240
      uo=157.73985;
      vo=134.19819;
      fx=391.54809;
      fy=395.45221;
    }
  };
  
 */ 
  
  struct options {  
    int feature,extract,match,outlier,solver; // options for feature points usage and solving methods
    options () { 
    //Default option values
    feature=1;
    extract=1;
    match=1;
    outlier=1;
    solver=1;
    }
  };
  
 
  // general parameters
  struct parameters {
    MonoVisualOdometry::options   option;           // options for feature usage
    //MonoVisualOdometry::calibration calib;          // camera calibration parameters
  };
  
  struct pose {  
    int N,iteration;
    float x_net,y_net,heading_net,Z_avg1,Z_avg2,run_time;  // net(absolute) Dx,Dy,phi wrt to start frame
    float x_rel,y_rel,heading_rel;	// relative Dx,Dy,phi wrt to previous frame
    pose () { 
    //Default option values
    N=0;
    iteration=0;
    x_net=0;
    y_net=0;
    heading_net=0;
    Z_avg1=0;
    Z_avg2=0;
    run_time=0;  
    x_rel=0;
    y_rel=0;
    heading_rel=0;  
    }
  };
  
  cv::Mat img1,img2; //old(1) and new(2) frames obtained from camera
  int nframes; // overall count of frames taken
  bool opticalFlow; // set(=1) for optical flow method
  
  // constructor, takes as input a parameter structure:
  MonoVisualOdometry (parameters param);
  
  // deconstructor
  ~MonoVisualOdometry ();
  
  //function to get image from ROS
  
  
  // find keypoints
  void findKeypoints();  
  
  // find descriptors
  void findDescriptor();    
  
  // find matches
  void findMatches();
  
  // find good_matches
  void findGoodMatches();
  
  // calc normalised 3D coordinates
  void calcNormCoordinates();
  
  // calc pose vector x_vect={Dx,Dy,phi,Z}
  void calcPoseVector();
  
  // update motion history
  void update_Motion();
  
  // calculate feature matches using optical flow
  void calcOpticalFlow();
  
  // run the entire process
  void run(); 
  
  // get output 
  void output(pose& );
  
protected:

    //gradient of error function for each variable
    float df_dDx(float Dx,float Dy, float phi, float Z, float **A, float **B, int N);
    float df_dDy(float Dx,float Dy, float phi, float Z, float **A, float **B, int N);
    float df_dphi(float Dx,float Dy, float phi, float Z, float **A, float **B, int N);
    float df_dZ(float Dx,float Dy, float phi, float Z, float **A, float **B, int N);        
    
    //ransacTest for outlier removal
    void ransacTest(const std::vector<cv::DMatch> matches,const std::vector<cv::KeyPoint>&keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& goodMatches,double distance,double confidence);


//(made public) int nframes; 	// overall count of frames taken
    float net_Dx,net_Dy,net_phi,net_Z1,net_Z2,Zsum,Rcos,Rsin; 	// net pose params at any instant (wrt start pose)
    float rel_Dx,rel_Dy,rel_phi;	// relative Dx,Dy,phi wrt to previous frame
//(made public) cv::Mat img1,img2; 	//old(1) and new(2) frames obtained from camera
    clock_t time; 	// variable to track time taken to run code
    float run_time;	//time for single run
    int N;	// no of good_matches obtained
    int count; 	//no of iterations for (solver ex gradient descent) convergence
    int feature,extract,match,outlier,solver; 	// options for feature points usage and solving methods
    float *u_old,*v_old,*u_new,*v_new; 	// old and new pixel coordinates (array)
    float **A,**B; 	//A(Nx2):old [X/Z Y/Z 1] & B(Nx2):new [Xn/Z Yn/Z 1]
    float Dx,Dy,phi,Z,Dx_o,Dy_o,phi_o,Z_o; 	// old and new camera translation and rotation params and depth Z
    float e; 	// error while solving minimization problem
    float gm; 	// param for controlling gradient descent/newton-raphson method
    cv::vector<cv::KeyPoint> keypoints1, keypoints2; 	//keypoints detected in two consecutive images 
    cv::vector<cv::Point2f> keypoints1_2f,keypoints2_2f;	//keypoints for calcOpticalFlow
    cv::Mat descriptors1, descriptors2; 	//descriptors calculated for the corresponding keypoints
    cv::vector<cv::DMatch> matches;      //matches among the keypoints
    cv::vector<cv::DMatch > good_matches;    //good_matches among the matches
    float fx;  // f/dx (in pixels)
    float fy;  // f/dy (in pixels)    
    float uo; // principal point (u-coordinate)
    float vo; // principal point (v-coordinate)
//    vector<uchar> status; // flag to check whether optical flow matching is found

private:
  //  parameters  param;     // common parameters
};

#endif // MONO_ODOMETRY_H	

  
  
  
