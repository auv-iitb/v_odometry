/*
 * This program is used to test the odometry.cpp code using live video feed from a pre-calibrated camera. Outputs are camera translation in x and y, rotation angle  and depth of points in the image (assuming it to be same for all points)
 * It also gives net pose {x-transl,y-transl,net heading} as the output.
 * 
 * Usage ex.: ./a.out 4 2 1 1 2
 */

#include <iostream>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;


float df_dDx(float Dx, float Dy, float phi, float Z, float **A, float **B, int N)
{
 float sum=0;
 for(int i=0;i<N;i++)
 {sum=sum+2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]));
 }
 return sum;
}

float df_dDy(float Dx,float Dy, float phi, float Z, float **A, float **B, int N)
{
 float sum=0;
 for(int i=0;i<N;i++)
 {sum=sum+2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
 }
 return sum;
}

float df_dphi(float Dx,float Dy, float phi, float Z, float **A, float **B, int N)
{
 float sum=0;
 for(int i=0;i<N;i++)
 {sum=sum + 2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) * ((-Z)*(-A[i][0]*sin(phi)-A[i][1]*cos(phi))) + 2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) * ((-Z)*(A[i][0]*cos(phi)-A[i][1]*sin(phi)));
 }
 return sum;
}

float df_dZ(float Dx,float Dy, float phi, float Z, float **A, float **B, int N)
{
 float sum=0;
 for(int i=0;i<N;i++)
 {sum=sum + 2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) * ((-1)*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) + 2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) * ((-1)*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
 }
 return sum;
}

void ransacTest(const std::vector<cv::DMatch> matches,const std::vector<cv::KeyPoint>&keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& goodMatches,double distance,double confidence)
{
    goodMatches.clear();
    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it)
    {
        // Get the position of old img keypoints
        float x= keypoints1[it->queryIdx].pt.x;
        float y= keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
        // Get the position of new img keypoints
        x= keypoints2[it->trainIdx].pt.x;
        y= keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x,y));
    }
    
    // Compute F matrix using RANSAC
    std::vector<uchar> inliers(points1.size(),0);
    cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2),inliers,FM_RANSAC,distance,confidence); // confidence probability
    // extract the surviving (inliers) matches
    std::vector<uchar>::const_iterator
    itIn= inliers.begin();
    std::vector<cv::DMatch>::const_iterator
    itM= matches.begin();
    // for all matches
    for ( ;itIn!= inliers.end(); ++itIn, ++itM)
    {
        if (*itIn)
        { // it is a valid match
            goodMatches.push_back(*itM);
        }
    }
}



int main(int argc, char** argv)
{
    VideoCapture cap(1); //1 - open the non-default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
        
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
            
    int ov_count=0;
    float net_Dx,net_Dy,net_phi,net_Z1,net_Z2,Zsum,Rcos,Rsin;
    net_Dx=0;net_Dy=0;net_phi=0;net_Z1=0;net_Z2=0;Zsum=0;
    Mat frame_old,frame;
    namedWindow("frames",1);
    for(int i=0;i<100;i++)
    {	
        cap >> frame; // get a new frame from camera
       	ov_count++;
        cvtColor(frame, frame, CV_BGR2GRAY);
        imshow("frames", frame);
        if(waitKey(5) >= 0) break;// waitKey will bring unrqrd delay
        
        if(ov_count>=2){
        // odometry.cpp code 
        // new image=frame & old image= frame_old

clock_t time;        
time=clock();
int N,count,feature,extract,match,outlier,solver;
float *u_old,*v_old,*u_new,*v_new;
float **A,**B;
float uo,vo,fx,fy,Z,Dx,Dy,phi,e,Dx_o,Dy_o,phi_o,Z_o,gm;

//Default option values
feature=1;
extract=1;
match=1;
outlier=1;
solver=1;
//Argument input for option selection
if (argc==6){
 feature=atoi(argv[1]);
 extract=atoi(argv[2]);
 match=atoi(argv[3]);
 outlier=atoi(argv[4]); 
 solver=atoi(argv[5]);
}
// Intrinsic Calibration parameters for img size 320x240
uo=157.73985;
vo=134.19819;
fx=391.54809;
fy=395.45221;

    Mat img1 = frame_old;
    Mat img2 = frame;
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints
    vector<KeyPoint> keypoints1, keypoints2;
    switch(feature)
    {
     case 1: //FAST
     {int threshold=110;
     FastFeatureDetector detector(threshold);
     detector.detect(img1, keypoints1);
     detector.detect(img2, keypoints2);
     break;
     }
     case 2: //SURF
     {SurfFeatureDetector detector(3000);
     detector.detect(img1, keypoints1);
     detector.detect(img2, keypoints2);
     break;
     }
     case 3: //GFTT
     {int maxCorners=200;
      GoodFeaturesToTrackDetector detector(maxCorners);
      detector.detect(img1, keypoints1);
      detector.detect(img2, keypoints2);
      break;
     }
     case 4: //ORB
     {int maxCorners=200;
      OrbFeatureDetector detector(maxCorners);
      detector.detect(img1, keypoints1);
      detector.detect(img2, keypoints2);     
      break;
     }
     case 5: //Harris  (change threshold, presently some default threshold)
     {
      Ptr<FeatureDetector> detector= FeatureDetector::create("HARRIS");
      detector->detect(img1, keypoints1);
      detector->detect(img2, keypoints2);      
     }     
    }
   
    // computing descriptors
    Mat descriptors1, descriptors2;
    switch(extract)
    {
     case 1: //SURF
     {
      SurfDescriptorExtractor extractor;
      extractor.compute(img1, keypoints1, descriptors1);
      extractor.compute(img2, keypoints2, descriptors2);
      break;
     }
     case 2: //SIFT
     {
      SiftDescriptorExtractor extractor;
      extractor.compute(img1, keypoints1, descriptors1);
      extractor.compute(img2, keypoints2, descriptors2);
      break;
     }
     case 3: //ORB
     {
      OrbDescriptorExtractor extractor;
      extractor.compute(img1, keypoints1, descriptors1);
      extractor.compute(img2, keypoints2, descriptors2);
      break;
     }     
    }
    
    // matching descriptors
    vector<DMatch> matches;
    switch (match)
    {
     case 1: //BruteForce
     {
     BFMatcher matcher(NORM_L2);
     matcher.match(descriptors1, descriptors2, matches);
     break;
     }
     case 2: //Flann
     {
     FlannBasedMatcher matcher;
     matcher.match(descriptors1, descriptors2, matches);
     break;
     }
    }
 
    // finding good matches
    vector< DMatch > good_matches; 
    switch (outlier)
    { 
     case 1:
     {
     double distance=40.; //quite adjustable/variable
     double confidence=0.99; //doesnt affect much when changed
     ransacTest(matches,keypoints1,keypoints2,good_matches,distance,confidence); 
     break;
     }
     case 2:
     {
     //look whether the match is inside a defined area of the image
     //only 25% of maximum of possible distance
     double tresholdDist = 0.25*sqrt(double(img1.size().height*img1.size().height + img1.size().width*img1.size().width));
     good_matches.reserve(matches.size());  
     for (size_t i = 0; i < matches.size(); ++i)
       {
        Point2f from = keypoints1[matches[i].queryIdx].pt;
        Point2f to = keypoints2[matches[i].trainIdx].pt;
        //calculate local distance for each possible match
        double dist = sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
        //save as best match if local distance is in specified area and on same height
        if (dist < tresholdDist)
          {
          good_matches.push_back(matches[i]);
          }
      }
     break;	
     }	
     case 3: //dist<2*min_dist
     {
        double max_dist = 0; double min_dist = 100;

 	 //-- Quick calculation of max and min distances between keypoints
 	 for( int i = 0; i < descriptors1.rows; i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
  	  }

	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	  //-- PS.- radiusMatch can also be used here.
	
	  for( int i = 0; i < descriptors1.rows; i++ )
	  { if( matches[i].distance < 2*min_dist )
	    { good_matches.push_back( matches[i]); }
	  }		
     }
    }

 matches=good_matches; // update matches by good_matches
 N=matches.size();  // no of matched feature points   

// Old and new consecutive frames pixel coordinate
u_old=new float [N]; 
v_old=new float [N];
u_new=new float [N];
v_new=new float [N];

A=new float* [N]; //old normalised coordinates [X/Z Y/Z 1]
B=new float* [N]; //new normalised coordinates [Xn/Z Yn/Z 1]

for(int i=0; i<N; i++) 
{
    A[i] = new float [3];
    B[i] = new float [3];
}

// Obtaining pixel coordinates and normalised 3D coordinates of feature points
for(size_t i = 0; i < N; i++)
{
    Point2f point1 = keypoints1[matches[i].queryIdx].pt;
    Point2f point2 = keypoints2[matches[i].trainIdx].pt;
    u_old[i]=point1.x;
    v_old[i]=point1.y;
    u_new[i]=point2.x;
    v_new[i]=point2.y;

    A[i][0] = (u_old[i]-uo)/fx; 
    A[i][1] = (v_old[i]-vo)/fy; 
    A[i][2] = 1;

    B[i][0] = (u_new[i]-uo)/fx;
    B[i][1] = (v_new[i]-vo)/fy;
    B[i][2] = 1;

}

// Finding least square error using Gradient-Descent or Newton-Raphson Method 
// x_vect={Dx,Dy,phi,Z} and x(n+1)=x(n)-grad(f(x(n)))
// f(x)=sum{i=1 to N}[(Dx-Z(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))^2] + sum{i=1 to N}[(Dy-Z(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))^2]
// grad(f(x))={df/dDx,df/dDy,df/dphi,df/dZ}

//initial guess
Dx=0;Dy=0;phi=0;Z=1; 

// Initial error
e=0;
for(size_t i = 0; i < N; i++){
 e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
}

// Iterate x_vect={Dx,Dy,phi,Z} using gradient functions until error<0.01
count=0;
//gm=0.005;
while(e>=0.01){
	count++;
//Old x_vect={Dx,Dy,phi,Z}
 Dx_o=Dx;Dy_o=Dy;phi_o=phi;Z_o=Z;
switch (solver)
{
 case 1: gm=0.005; // Gradient Descent
 break;
 case 2: gm=1/e; // Newton-Raphson
 break;
}
 
//New x_vect={Dx,Dy,phi,Z}
 Dx=Dx_o-gm*df_dDx(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
 Dy=Dy_o-gm*df_dDy(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
 phi=phi_o-gm*df_dphi(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
 Z=Z_o-gm*df_dZ(Dx_o,Dy_o,phi_o,Z_o,A,B,N);

// Find error
 e=0;
 for(size_t i = 0; i < N; i++){
 e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
 }
//cout<<e<<"\t";
}

time=clock()-time;
cout<<"N="<<N<<"\t"<<"Dx="<<Dx<<"\t"<<"Dy="<<Dy<<"\t"<<"phi="<<phi<<"\t"<<"Z="<<Z<<"\t";
cout<<"e="<<e<<"\t"<<"iteratn="<<count<<"\t";
cout<<"time="<<((float)time)/CLOCKS_PER_SEC<<"\n";

// net pose calculation (wrt starting pose) 
	Rcos=Dx*cos(phi)+Dy*sin(phi);
	Rsin=Dx*sin(phi)-Dy*cos(phi);	
	net_Dx=net_Dx+Rcos*cos(net_phi)-Rsin*sin(net_phi); //net camera translation in x-direction wrt to starting pose
	net_Dy=net_Dy+Rcos*sin(net_phi)+Rsin*cos(net_phi); //net camera translation in y-direction wrt to starting pose	
        net_phi=net_phi+phi; 				   //net heading angle (anti-clk +ve)
        Zsum=Zsum+Z;					   
        net_Z1=Zsum/(ov_count-1);			   //average estimated_1 value of depth of ground from camera
	  if(ov_count==2) net_Z2=Z;
          else net_Z2=(net_Z2+Z)/2;			   //average estimated_2 value of depth of ground from camera
	cout<<"Dx_net="<<net_Dx<<"\t"<<"Dy_net="<<net_Dy<<"\t"<<"phi_net="<<net_phi<<"\t"<<"Z_net1="<<net_Z1<<"\t";
	cout<<"Z_net2="<<net_Z2<<"\n"<<"reso"<<frame.size()<<"\n";
        }
    	frame_old=frame.clone();        
        imshow("frames_old", frame_old);    	
        if(waitKey(5) >= 0) break;// waitKey will bring unrqrd delay        
    }
    cap.release();
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

