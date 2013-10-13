#include "mono_odometry.h"

using namespace std;
using namespace cv;

MonoVisualOdometry::MonoVisualOdometry (parameters param) {
      net_Dx=0;net_Dy=0;net_phi=0;net_Z1=0;net_Z2=0;Zsum=0; //pose initialisation
      // calib parameters
      uo=157.73985;
      vo=134.19819;
      fx=391.54809;
      fy=395.45221;
      // getting feature options from user
      feature=param.option.feature;
      extract=param.option.extract;
      match=param.option.match;
      outlier=param.option.outlier;
      solver=param.option.solver;      
}

MonoVisualOdometry::~MonoVisualOdometry () {
  //delete matcher;
}

void MonoVisualOdometry::findKeypoints() { 
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
}

void MonoVisualOdometry::findDescriptor() {
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
}

void MonoVisualOdometry::findMatches() {
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
}

void MonoVisualOdometry::findGoodMatches() {
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
     case 3: //dist<2*min_dist
     {
        double max_dist = 0; double min_dist = 100;

 	 //-- Quick calculation of max and min distances between keypoints
 	 for( int i = 0; i < descriptors1.rows; i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
  	  }

	  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	  //-- PS.- radiusMatch can also be used here.
	
	  for( int i = 0; i < descriptors1.rows; i++ )
	  { if( matches[i].distance < 2*min_dist )
	    { good_matches.push_back( matches[i]); }
	  }
	  break;		
     }
    }
    matches=good_matches; // update matches by good_matches    
    N=matches.size();  // no of matched feature points
}

void MonoVisualOdometry::calcOpticalFlow(){
    int maxCorners=180;
    GoodFeaturesToTrackDetector detector(maxCorners);
    detector.detect(img1, keypoints1);
    
    // convert KeyPoint to Point2f
    for (int i=0;i<keypoints1.size(); i++)
       {
        float x= keypoints1[i].pt.x;
        float y= keypoints1[i].pt.y;
        keypoints1_2f.push_back(cv::Point2f(x,y));
       }
    // LK Sparse Optical Flow   
    vector<uchar> status; 
    vector<float> err;
    Size winSize=Size(21,21);
    int maxLevel=3;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    int flags=0;
    double minEigThreshold=1e-4;
    cv::calcOpticalFlowPyrLK(img1, img2, keypoints1_2f, keypoints2_2f, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);
    N=keypoints2_2f.size();  // no of matched feature points
}

void MonoVisualOdometry::calcNormCoordinates() {
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
     	 Point2f point1,point2;
         if(!opticalFlow){
         point1 = keypoints1[matches[i].queryIdx].pt;
         point2 = keypoints2[matches[i].trainIdx].pt;
         }
         else {
         point1 = keypoints1_2f[i];
         point2 = keypoints2_2f[i];         
         }
         u_old[i]=point1.x;
         v_old[i]=point1.y;
         u_new[i]=point2.x;
         v_new[i]=point2.y;

         A[i][0] = -(u_old[i]-uo)/fx; 
         A[i][1] = (v_old[i]-vo)/fy; 
         A[i][2] = 1;

         B[i][0] = -(u_new[i]-uo)/fx;
         B[i][1] = (v_new[i]-vo)/fy;
         B[i][2] = 1;

     }
}

void MonoVisualOdometry::calcPoseVector() {
    // Finding least square error using Gradient-Descent or Newton-Raphson Method 
    // x_vect={Dx,Dy,phi,Z} and x(n+1)=x(n)-grad(f(x(n)))
    // f(x)=sum{i=1 to N}[(Dx-Z(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))^2] + sum{i=1 to N}[(Dy-Z(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))^2]
    // grad(f(x))={df/dDx,df/dDy,df/dphi,df/dZ}
    
    //initial guess
    Dx=0.001;Dy=0.001;phi=0.1;Z=1.5; 

    // Initial error
    e=0;
    for(size_t i = 0; i < N; i++){
        e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
    }

    // Iterate x_vect={Dx,Dy,phi,Z} until error<0.01
    count=0; 	//no of iterations for error to converge
    
    while(e>=0.01){
	count++;
        //Old x_vect={Dx,Dy,phi,Z}
        Dx_o=Dx;Dy_o=Dy;phi_o=phi;Z_o=Z;
        switch (solver)
        {
         case 1: gm=0.005; // Gradient-Descent
         break;
         case 2: gm=1/e; // Newton-Raphson
         break;
        }
 
        //New x_vect={Dx,Dy,phi,Z}
        Dx = Dx_o - gm*df_dDx(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
        Dy = Dy_o - gm*df_dDy(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
        phi = phi_o - gm*df_dphi(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
        Z = Z_o - gm*df_dZ(Dx_o,Dy_o,phi_o,Z_o,A,B,N);

	// Find error
	e=0;
	for(size_t i = 0; i < N; i++){
	    e = e + (Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
	}
    }
    
}

void MonoVisualOdometry::update_Motion(){
    // net pose calculation (wrt starting pose) 
    Rcos=Dx*cos(phi)+Dy*sin(phi);
    Rsin=Dx*sin(phi)-Dy*cos(phi);	
    net_Dx=net_Dx+Rcos*cos(net_phi)-Rsin*sin(net_phi); //net camera translation in x-direction wrt to starting pose
    net_Dy=net_Dy+Rcos*sin(net_phi)+Rsin*cos(net_phi); //net camera translation in y-direction wrt to starting pose	
    net_phi=net_phi+phi; 			       //net heading angle (anti-clk +ve)
    Zsum=Zsum+Z;					   
    net_Z1=Zsum/(nframes-1);			       //average estimated_1 value of depth of ground from camera
    if(nframes==2) net_Z2=Z;
    else net_Z2=(net_Z2+Z)/2;			       //average estimated_2 value of depth of ground from camera
} 

void MonoVisualOdometry::run() {
    // get image frames

    // start the timer
    time=clock();    
    
    // convert to grayscale
    cvtColor(img1,img1,CV_BGR2GRAY);
    cvtColor(img2,img2,CV_BGR2GRAY);  
    
  if(!opticalFlow){
    // find keypoints
    findKeypoints();  
  
    // find descriptors
    findDescriptor();    
  
    // find matches
    findMatches();
  
    // find good_matches
    findGoodMatches();
  }
  else {
    //calculate matched feature points optical flow
    calcOpticalFlow();
  }
    // calc normalised 3D coordinates
    calcNormCoordinates();
  
    // calc pose vector x_vect={Dx,Dy,phi,Z}
    calcPoseVector();
  
    // update motion history
    update_Motion();
    
    time=clock()-time;
    run_time=((float)time)/CLOCKS_PER_SEC;   //time for single run
/*
    // display the two frames
    imshow("Old frame", img1);
    if(waitKey(5) >= 0) break;// waitKey will bring unrqrd delay  
    imshow("New frame", img2);
    if(waitKey(5) >= 0) break;// waitKey will bring unrqrd delay      
*/
}

//void MonoVisualOdometry::output(int N,float x_net,float y_net,float heading_net,float Z_avg1,float Z_avg2,int iteration,float run_time) {
void MonoVisualOdometry::output(pose& position) {
    position.N=N;
    position.iteration=count;
    position.x_net=net_Dx;
    position.y_net=net_Dy;
    position.heading_net=net_phi;
    position.Z_avg1=net_Z1;
    position.Z_avg2=net_Z2;
    position.run_time=run_time; 
}

float MonoVisualOdometry::df_dDx(float Dx,float Dy, float phi, float Z, float **A, float **B, int N) {
  float sum=0;
  for(int i=0;i<N;i++)
  {
    sum=sum+2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]));
  }
  return sum;
}

float MonoVisualOdometry::df_dDy(float Dx,float Dy, float phi, float Z, float **A, float **B, int N) {
  float sum=0;
  for(int i=0;i<N;i++)
  {
    sum=sum+2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
  }
  return sum;
}

float MonoVisualOdometry::df_dphi(float Dx,float Dy, float phi, float Z, float **A, float **B, int N) {
  float sum=0;
  for(int i=0;i<N;i++)
  {
    sum=sum + 2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) * ((-Z)*(-A[i][0]*sin(phi)-A[i][1]*cos(phi))) + 2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) * ((-Z)*(A[i][0]*cos(phi)-A[i][1]*sin(phi)));
  }
  return sum;
}

float MonoVisualOdometry::df_dZ(float Dx,float Dy, float phi, float Z, float **A, float **B, int N) {
  float sum=0;
  for(int i=0;i<N;i++)
  {
    sum=sum + 2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) * ((-1)*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) + 2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) * ((-1)*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
  }
  return sum;
}

void MonoVisualOdometry::ransacTest(const std::vector<cv::DMatch> matches,const std::vector<cv::KeyPoint>&keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& goodMatches,double distance,double confidence)
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

