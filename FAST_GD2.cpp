#include <iostream>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

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


int main(int argc, char** argv)
{
clock_t time;
time=clock();
int N,count;
float *u_old,*v_old,*u_new,*v_new;
float *X_old,*Y_old,*X_new,*Y_new;
float *P,*K;
float **A,**B;
float uo,vo,fx,fy,Z,Dx,Dy,phi,e,Dx_o,Dy_o,phi_o,Z_o,gm;
float Tr[3][3];


// Intrinsic Calibration parameters for img size 320x240
uo=157.73985;
vo=134.19819;
fx=391.54809;
fy=395.45221;

    if(argc != 3)
    {
        help();
        return -1;
    }


    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints
    FastFeatureDetector detector(130);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);


    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

N=matches.size();  // no of matched feature points
// Old and new consecutive frames pixel coordinate
u_old=new float [N]; 
v_old=new float [N];
u_new=new float [N];
v_new=new float [N];
// Old and new consecutive frames 3D coordinate
X_old=new float [N]; 
Y_old=new float [N];
X_new=new float [N];
Y_new=new float [N];
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

//initial guess
Dx=0;Dy=0;phi=0;Z=1; 

// Initial error
e=0;
for(size_t i = 0; i < N; i++){
 e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
}

// Iterate x_vect={Dx,Dy,phi,Z} using gradient functions until error<0.01 or count>100
count=0;
gm=0.005; //optimum gm=0.005, total_program_time=0.05 s, e>=0.01, only 1 common gradient descent
while(e>=0.01){
	count++;
//Old x_vect={Dx,Dy,phi,Z}
 Dx_o=Dx;Dy_o=Dy;phi_o=phi;Z_o=Z;

 //Iterate only Z
 //Dx_o=Dx;Dy_o=Dy;phi_o=phi;
 Z=Z_o-gm*df_dZ(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
 Z_o=Z;
 
 int icount=0;
 while(e>=0.01&&icount<100){ icount++;
  Dx_o=Dx;Dy_o=Dy;phi_o=phi;
  //Iterate over Dx,Dy,phi keeping Z const
  Dx=Dx_o-gm*df_dDx(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
  Dy=Dy_o-gm*df_dDy(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
  phi=phi_o-gm*df_dphi(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
  // Find error
  e=0;
  for(size_t i = 0; i < N; i++){
  e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
  }
 }
 
// Find error
 e=0;
 for(size_t i = 0; i < N; i++){
 e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
 }
//cout<<e<<"\t";
}

time=clock()-time;
cout<<N<<"\n"<<Dx<<"\n"<<Dy<<"\n"<<phi<<"\n"<<Z<<"\n";
cout<<e<<"\n"<<count<<"\n";
cout<<((float)time)/CLOCKS_PER_SEC<<"\n";

/*
    // drawing the rmatches
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);
*/
    return 0;
}
