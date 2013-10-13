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

using namespace std;
using namespace cv;

static void help()
{
    printf("Usage:\n ./a.out <image1> <image2>\n");
}

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
        // Get the position of left keypoints
        float x= keypoints1[it->queryIdx].pt.x;
        float y= keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
        // Get the position of right keypoints
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
clock_t time;
time=clock();
int N,count,feature,extract,match,outlier,solver;
float *u_old,*v_old,*u_new,*v_new;
float **A,**B;
float uo,vo,fx,fy,Z,Dx,Dy,phi,e,Dx_o,Dy_o,phi_o,Z_o,gm;

    if(argc < 3)
    { help();
      return -1;
    }

//Default option values
feature=1;
extract=1;
match=1;
outlier=1;
solver=1;
//Argument input for option selection
if (argc>=8){
 feature=atoi(argv[3]);
 extract=atoi(argv[4]);
 match=atoi(argv[5]);
 outlier=atoi(argv[6]); 
 solver=atoi(argv[7]);
}
// Intrinsic Calibration parameters for img size 320x240
uo=157.73985;
vo=134.19819;
fx=391.54809;
fy=395.45221;

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
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
     {int threshold=130;
     FastFeatureDetector detector(threshold);
     detector.detect(img1, keypoints1);
     detector.detect(img2, keypoints2);
     break;
     }
     case 2: //SURF
     {SurfFeatureDetector detector(2000);
     detector.detect(img1, keypoints1);
     detector.detect(img2, keypoints2);
     break;
     }
     case 3: //GFTT
     {int maxCorners=150;
      GoodFeaturesToTrackDetector detector(maxCorners);
      detector.detect(img1, keypoints1);
      detector.detect(img2, keypoints2);
      break;
     }
     case 4: //ORB
     {int maxCorners=150;
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
     double distance=50.; //quite adjustable/variable
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
    }

 matches=good_matches; // update matches by good_matches
 N=matches.size();  // no of matched feature points   

// estimateRigidTransform method to find rotation
std::vector<Point2f> src;
std::vector<Point2f> dst;
Point2f point_1,point_2;
Point2f centre(uo,vo);

for(size_t i = 0; i < N; i++)
{
    point_1 = keypoints1[matches[i].queryIdx].pt;
    point_2 = keypoints2[matches[i].trainIdx].pt;  
    point_1 = point_1-centre;
    point_2 = point_2-centre;
   src.push_back(point_1);
   dst.push_back(point_2);    
}
Mat rot=estimateRigidTransform(src,dst,false);
cout<<rot<<"\n";

 
// Old and new consecutive frames pixel coordinate
u_old=new float [N]; 
v_old=new float [N];
u_new=new float [N];
v_new=new float [N];

A=new float* [N]; //old [X/Z Y/Z 1]
B=new float* [N]; //new [Xn/Z Yn/Z 1]

for(int i=0; i<N; i++) 
{
    A[i] = new float [3];
    B[i] = new float [3];
}

// Obtaining pixel coordinates of feature points
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

// Finding least square error using Gradient Descent Method 
// x_vect={Dx,Dy,phi,Z} and x(n+1)=x(n)-grad(f(x(n)))
// f(x)=sum{i=1 to N}[(Dx-Z(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))^2] + sum{i=1 to N}[(Dy-Z(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))^2]
// grad(f(x))={df/dDx,df/dDy,df/dphi,df/dZ}

//Fix Dx,Dy,Z as only ROTATION case
Dx=0;Dy=0;Z=1;
//initial guess (for phi alone)
phi=0.2;

// Initial error
e=0;
for(size_t i = 0; i < N; i++){
 e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
}

// Iterate over phi using gradient functions until error<0.01
count=0;
float e_old=0;
//gm=0.005;
while(e>=0.01&&e_old!=e){
	count++;
	e_old=e;
//Old phi
 phi_o=phi;
switch (solver)
{
 case 1: gm=0.01; // Gradient Descent
 break;
 case 2: gm=1/e; // Newton-Raphson
 break;
}
 
//New phi
 phi=phi_o-gm*df_dphi(Dx,Dy,phi_o,Z,A,B,N);
 
// Find error
 e=0;
 for(size_t i = 0; i < N; i++){
 e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
 }
cout<<e<<"\t";
}

time=clock()-time;
cout<<N<<"\n"<<Dx<<"\n"<<Dy<<"\n"<<phi<<"\n"<<Z<<"\n";
cout<<e<<"\n"<<count<<"\n";
cout<<((float)time)/CLOCKS_PER_SEC<<"\n";



    // drawing the rmatches
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}
