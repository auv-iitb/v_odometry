#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "mono_odometry.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    
    MonoVisualOdometry::parameters param;
    param.option.feature=4;
    param.option.extract=2;
    param.option.match=1;
    param.option.outlier=1;
    param.option.solver=1; 
    
    // set overall count; input image; getoutput
    Mat frame_old,frame;
    frame_old=imread("1.png");
    frame=imread("2.png");
    
    if(frame_old.empty() || frame.empty())
    {
        cout<<"Can't read one of the images\n";
        return -1;
    }
    
    MonoVisualOdometry odom(param);               
    odom.nframes=1;
    
    for(int i=0 ;i<1 ;i++ ){
	// get new frame
        //cin>>frame;
        odom.nframes++;

        if(odom.nframes>=2) {
  	  // run odometry
  	  odom.img1=frame_old;
  	  odom.img2=frame;
   	  odom.run();
   	  MonoVisualOdometry::pose position;
	  odom.output(position);
	  cout<<position.N<<"\n";
	  cout<<position.x_net<<"\n";
    	}
	
	//copy the frame to frame_old
	frame_old=frame.clone();    	
    }
    cout<<"Success\n";
    return 0;
}
