#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "mono_odometry.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // class input parameters - feature options
    MonoVisualOdometry::parameters param;
/* not needed for optical flow method

    param.option.feature=4;
    param.option.extract=2;
    param.option.match=1;
    param.option.outlier=1;
*/
    param.option.method=1;
    param.option.solver=1; 
        
    Mat frame_old,frame;
    frame_old=imread(argv[1]); // for testing
    frame=imread(argv[2]); // for testing
    
    if(frame_old.empty() || frame.empty())
    {
        cout<<"Can't read one of the images\n";
        return -1;
    }
    
    MonoVisualOdometry odom(param);               
    odom.nframes=1; // keeps track of overall frames count
//    odom.nframes=0; // shud be intitialised with =0; for testing =1;
    odom.opticalFlow=true; 	//necessary for optical flow
    
//    while(!flag){	//actual
    for(int i=0 ;i<1 ;i++ ){	// for(;;) testing
	// get new frame from ROS
        //cin>>frame;  
        odom.nframes++;

        if(odom.nframes>=2) {
  	  // run odometry
  	  odom.img1=frame_old;
  	  odom.img2=frame;
   	  odom.run(); 	// run the main odometry calculations
   	  MonoVisualOdometry::pose position; //output struct
	  odom.output(position);  // get output parameters
	  cout<<"N="<<position.N<<"\n"; // no of good_matches
	  cout<<"x_net="<<position.x_net<<"\n"; // net x-translation
	  cout<<"y_net="<<position.y_net<<"\n"; // net y-translation
	  cout<<"heading_net="<<position.heading_net<<"\n"; // net heading
	  cout<<"Z_1="<<position.Z_avg1<<"\n"; // avg Z estimation 1
	  cout<<"Z_2="<<position.Z_avg2<<"\n"; // avg Z estimation 2
	  cout<<"iters="<<position.iteration<<"\n"; // no of solver iterations
  	  cout<<"time="<<position.run_time<<"\n"; // .run() time
    	  cout<<"x_rel="<<position.x_rel<<"\n";  // relative x-translation
    	  cout<<"y_rel="<<position.y_rel<<"\n";  // relative y-translation
          cout<<"heading_rel="<<position.heading_rel<<"\n";  	  // relative heading change
          cout<<"rot matrix="<<position.rot<<"\n";		//transform estimated using estimateRigidTransform
          cout<<"x_scaled="<<position.x_scaled<<"\n";		// scaled x-translation
          cout<<"y_scaled="<<position.y_scaled<<"\n";		// scaled y-translation
          cout<<"converged error="<<position.error<<"\n";
    	}
	
	//copy the frame to frame_old
	frame_old=frame.clone();    	
    }
    
    // ROS Plot x_net, y_net, heading_net wrt time
    
    return 0;
}
