#include <iostream>
#include <stdio.h>
#include "Eigen/Core"
#include "Eigen/LU"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

// Used Eigen 3.2.0 Library for matrix inversion [http://eigen.tuxfamily.org/index.php?title=Main_Page]
// DONT USE 2x2 adjoint/inverse function of Eigen library, its incorrect
// Problems: (1) Matrix calc computationally expensive
//           (2) Error diverging

using namespace std;
using namespace cv;

static void help()
{
    printf("Usage:\n ./a.out <image1> <image2>\n");
}

int main(int argc, char** argv){

int N,count;
float *u_old,*v_old,*u_new,*v_new;
float *X_old,*Y_old,*X_new,*Y_new;
float *P,*K;
float **A,**B;
float uo,vo,fx,fy,Z,Dx,Dy,cphi,sphi,e,num,den;
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
    FastFeatureDetector detector(190);
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
Eigen::MatrixXf A_mat(N,3),B_mat(N,3),Tr_mat(3,3),tmp_mat(3,3);

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

    A[i][0] = -(u_old[i]-uo)/fx; A_mat(i,0)=A[i][0];
    A[i][1] = (v_old[i]-vo)/fy; A_mat(i,1)=A[i][1];
    A[i][2] = 1; A_mat(i,2)=A[i][2];

    B[i][0] = -(u_new[i]-uo)/fx; B_mat(i,0)=B[i][0];
    B[i][1] = (v_new[i]-vo)/fy; B_mat(i,1)=B[i][1];
    B[i][2] = 1; B_mat(i,2)=B[i][2];

}

tmp_mat=A_mat.transpose()*A_mat;
Tr_mat=tmp_mat.inverse()*(A_mat.transpose())*B_mat;

//cout<<N<<"\n"<<A<<"\n"<<*A<<"\n"<<**A; 
//cout<<Tr_mat<<"\n";

// Transformation Matrix calculation
P=new float [N]; 
K=new float [N];
// Tr=inv(A'*A)*A'*B;
 cphi=(Tr_mat(0,0)+Tr_mat(1,1))/2;
 sphi=(Tr_mat(0,1)-Tr_mat(1,0))/2;

for(size_t i = 0; i < N; i++){
P[i]=cphi*A_mat(i,0)-sphi*A_mat(i,1)-B_mat(i,0); //(Dx/Z)
K[i]=sphi*A_mat(i,0)+cphi*A_mat(i,1)-B_mat(i,1);//(Dy/Z)
}

// Finding least error Z, Dx, Dy
Z=2; //initial guess (metres)
Dx=-Z*Tr_mat(2,0);
Dy=-Z*Tr_mat(2,1);

e=0;
for(size_t i = 0; i < N; i++){
 e=e+(Dx-Z*P[i])*(Dx-Z*P[i])+(Dy-Z*K[i])*(Dy-Z*K[i]);
}
count=0;
while(e>=0.1&&count<50){
	count++;
//cout<<Dx<<"\t";

// Calculate Z_opt
	num=0;den=0;
	for(size_t i = 0; i < N; i++){
	num=num+Dx*P[i]+Dy*K[i];
	den=den+P[i]*P[i]+K[i]*K[i];
	}
	Z=num/den;// New Z is Z_opt (min error e) and new Dx, Dy
	Dx=-Z*Tr_mat(2,0);
	Dy=-Z*Tr_mat(2,1);
// Find error
	e=0;
	for(size_t i = 0; i < N; i++){
	e=e+(Dx-Z*P[i])*(Dx-Z*P[i])+(Dy-Z*K[i])*(Dy-Z*K[i]);
	}
//cout<<e<<"\t";
}

//cout<<"\n"<<e<<"\t"<<Dx<<"\t"<<Dy<<"\t"<<Z<<"\n";

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
