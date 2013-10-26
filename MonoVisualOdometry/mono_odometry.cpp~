#include "mono_odometry.h"
#include <iostream>

using namespace std;
using namespace cv;

#define min_N 10	// min no of feature check for result reliability
#define min_phi 0.001   // radians
#define max_phi 0.3     // radians

MonoVisualOdometry::MonoVisualOdometry (parameters param) {
      net_Dx=0;net_Dy=0;net_phi=0;net_Z1=0;net_Z2=0;Zsum=0; //pose initialisation
      // calib parameters
      uo=param.calib.uo;
      vo=param.calib.vo;
      fx=param.calib.fx;
      fy=param.calib.fy;
      // getting feature options from user
      feature=param.option.feature;
      extract=param.option.extract;
      match=param.option.match;
      outlier=param.option.outlier;
      method=param.option.method;            
      solver=param.option.solver;
      mask=imread("mask_e.png",0);
}

MonoVisualOdometry::~MonoVisualOdometry () {
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
    int maxCorners=200;
    std::vector<cv::KeyPoint> _keypoints1;	// all keypoints detected
    GoodFeaturesToTrackDetector detector(maxCorners);
    detector.detect(img1, _keypoints1, mask);
    
    // convert KeyPoint to Point2f
    for (int i=0;i<_keypoints1.size(); i++)
       {
        float x= _keypoints1[i].pt.x;
        float y= _keypoints1[i].pt.y;
        keypoints1_2f.push_back(cv::Point2f(x,y));
       }
       
    // subpixel corner refinement for keypoints1_2f
    Size SPwinSize = Size(3,3);		//search window size=(2*n+1,2*n+1)
    Size zeroZone = Size(1,1);	// dead_zone size in centre=(2*n+1,2*n+1)
    TermCriteria SPcriteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    cornerSubPix(img1, keypoints1_2f, SPwinSize, zeroZone, SPcriteria);
       
    // LK Sparse Optical Flow   
    vector<uchar> status;
    vector<float> err;
    Size winSize=Size(21,21);
    int maxLevel=3;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    int flags=0;
    double minEigThreshold=1e-4;
    cv::calcOpticalFlowPyrLK(img1, img2, keypoints1_2f, keypoints2_2f, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);

    // subpixel corner refinement for keypoints2_2f
//    cornerSubPix(img2, keypoints2_2f, SPwinSize, zeroZone, SPcriteria);
        
    float dist;
    // convert Point2fs to KeyPoints
    for (int i=0;i<keypoints2_2f.size(); i++)
       {
        float x1= keypoints1_2f[i].x;
        float y1= keypoints1_2f[i].y;       	
        float x2= keypoints2_2f[i].x;
        float y2= keypoints2_2f[i].y;

        dist = sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) ); 	// dist betwn obtained matches
        if(status[i]==1 && dist<=20){			// min dist threshold
          KeyPoint kp1(x1,y1,1.0,-1.0,0.0,0,-1);
          keypoints1.push_back(kp1);
        
          KeyPoint kp2(x2,y2,1.0,-1.0,0.0,0,-1);        
          keypoints2.push_back(kp2);  
          fmatches.push_back(i); 
        }
       }

    N=keypoints2.size();  // no of matched feature points
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
         if(opticalFlow){
         point1 = keypoints1[i].pt;
         point2 = keypoints2[i].pt;         
         }
         else {
         point1 = keypoints1[matches[i].queryIdx].pt;
         point2 = keypoints2[matches[i].trainIdx].pt;         
         }
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
}

void MonoVisualOdometry::estimateTransformMatrix() {
     std::vector<Point2f> src;
     std::vector<Point2f> dst;
     Point2f point_1,point_2;
     Point2f centre(uo,vo);

     for(size_t i = 0; i < N; i++)
     {
         if(opticalFlow){
         point_1 = keypoints1[i].pt;
         point_2 = keypoints2[i].pt;         
         }
         else {
         point_1 = keypoints1[matches[i].queryIdx].pt;
         point_2 = keypoints2[matches[i].trainIdx].pt;         
         } 
         point_1 = point_1-centre;
         point_2 = point_2-centre;
         src.push_back(point_1);
         dst.push_back(point_2);    
     }
     rot=cv::estimateRigidTransform(src,dst,false);
     
     double cost=rot.at<double>(0,0);
     double sint=rot.at<double>(1,0);

     // remove abs(cost)>1 and abs(sint)>1
     if (cost>1.0) cost=1.0;
     if (cost<-1.0) cost=-1.0;     	
     if (sint>1.0) sint=1.0;
     if (sint<-1.0) sint=-1.0; 
     
     if (sint>0) {    	
     	phi=( acos(cost) + asin(sint) )/2.0;
     }
     else {
     	phi=( -acos(cost) + asin(sint) )/2.0;
     }
     tx=rot.at<double>(0,2);
     ty=rot.at<double>(1,2);    
    
     if (abs(phi)<=min_phi) {
     	phi=0; 		// to remove accumulation of small 0 error
     }
     if (abs(phi)>=max_phi) {	// to remove impractical values
     	if (phi>0) phi=max_phi; 	
     	else phi=-max_phi; 	
     }
     if (abs(phi)>=max_phi || N<=min_N) phi_status=false; 	// min 10 features change phi_status flag
     else phi_status=true;      
}

void MonoVisualOdometry::rotationScaledTranslation() {
    // Finding least square error using Gradient-Descent or Newton-Raphson Method 
    // x_vect={tx(=Dx/Z),ty(=Dy/Z),phi} and x(n+1)=x(n)-grad(f(x(n)))
    // f(x)=sum{i=1 to N}[(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))^2] + sum{i=1 to N}[(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))^2]
    // grad(f(x))={df/dtx,df/dty,df/dphi}
    
    //initial guess
    tx=0.001;ty=0.001;phi=0;

    // Initial error
    e=0;
    for(size_t i = 0; i < N; i++){
        e =e+(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
    }

    // Iterate x_vect={tx,ty,phi} until error<0.01
    count=0; 	//no of iterations for error to converge
    float e_old=0;
    float grad_sum=10;	// sum of squares of gradients
    
    while((e>=0.0001)&&(count<100)&&(grad_sum>=0.0001)){
	count++;
	e_old=e;
        //Old x_vect={tx,ty,phi}
        tx_o=tx;ty_o=ty;phi_o=phi;
        switch (solver)
        {
         case 1: gm=0.005; // Gradient-Descent
         break;
         case 2: gm=1/e; // Newton-Raphson
         break;
        }
 
        //New x_vect={tx,ty,phi}
        tx = tx_o - gm*df_dDx(tx_o,ty_o,phi_o,1,A,B,N);
        ty = ty_o - gm*df_dDy(tx_o,ty_o,phi_o,1,A,B,N);
        phi = phi_o - gm*df_dphi(tx_o,ty_o,phi_o,1,A,B,N);

	float dDx=df_dDx(tx_o,ty_o,phi_o,1,A,B,N);
	float dDy=df_dDy(tx_o,ty_o,phi_o,1,A,B,N);
	float dphi=df_dphi(tx_o,ty_o,phi_o,1,A,B,N);	
	grad_sum=dDx*dDx + dDy*dDy + dphi*dphi;

	// Find error
	e=0;
	for(size_t i = 0; i < N; i++){
	    e = e + (tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
	}
    }

    if (abs(phi)<=min_phi) {
    	phi=0; 		// to remove accumulation of small 0 error
    }
    if (abs(phi)>=max_phi) {	// to remove impractical values
    	if (phi>0) phi=max_phi; 	
    	else phi=-max_phi; 	
    }
    if (abs(phi)>=max_phi || N<=min_N) phi_status=false; 	// min 10 features change phi_status flag
    else phi_status=true;
        
}

void MonoVisualOdometry::rotationScaledTranslation_reg() {
    // Finding least square error using Gradient-Descent or Newton-Raphson Method 
    // x_vect={tx(=Dx/Z),ty(=Dy/Z),phi} and x(n+1)=x(n)-grad(f(x(n)))
    // f(x)=sum{i=1 to N}[(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))^2] + sum{i=1 to N}[(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))^2] + lam*(df_dphi())*(df_dphi());
    // grad(f(x))={df/dtx,df/dty,df/dphi}
    
    //initial guess
    tx=0.001;ty=0.001;phi=0;lam=0.005;

    float dDx,dDy,dphi,dphi_new; //gradients
    dphi=df_dphi(tx,ty,phi,1,A,B,N);    
    
    // Initial error
    e=0;
    for(size_t i = 0; i < N; i++){
        e =e+(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) + lam*(dphi*dphi);
    }

    // Iterate x_vect={tx,ty,phi} until error<0.01
    count=0; 	//no of iterations for error to converge
    float e_old=0;
    float grad_sum=10;	// sum of squares of gradients
    
    while((e>=0.0001)&&(count<100)&&(grad_sum>=0.0001)){
	count++;
	e_old=e;
        //Old x_vect={tx,ty,phi}
        tx_o=tx;ty_o=ty;phi_o=phi;
        switch (solver)
        {
         case 1: gm=0.005; // Gradient-Descent
         break;
         case 2: gm=1/e; // Newton-Raphson
         break;
        }

	dDx=df_dDx(tx_o,ty_o,phi_o,1,A,B,N);
	dDy=df_dDy(tx_o,ty_o,phi_o,1,A,B,N);
	dphi=df_dphi(tx_o,ty_o,phi_o,1,A,B,N);
	dphi_new=dphi/(1-2*lam*d2f_d2phi(tx_o,ty_o,phi_o,1,A,B,N));	
		
	grad_sum=dDx*dDx + dDy*dDy + dphi_new*dphi_new;
	 
        //New x_vect={tx,ty,phi}
        tx = tx_o - gm*dDx;
        ty = ty_o - gm*dDy;
        phi = phi_o - gm*dphi_new;

	dphi=df_dphi(tx,ty,phi,1,A,B,N);
	// Find error
	e=0;
	for(size_t i = 0; i < N; i++){
	    e = e + (tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(tx-(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(ty-(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) + lam*(dphi*dphi);
	}
    }
    
    if (abs(phi)<=min_phi) {
    	phi=0; 		// to remove accumulation of small 0 error
    }
    if (abs(phi)>=max_phi) {	// to remove impractical values
    	if (phi>0) phi=max_phi; 	
    	else phi=-max_phi; 	
    }
    if (abs(phi)>=max_phi || N<=min_N) phi_status=false; 	// min 10 features change phi_status flag
    else phi_status=true;
}

void MonoVisualOdometry::rotationActualTranslation() {
    // Finding least square error using Gradient-Descent or Newton-Raphson Method 
    // x_vect={Dx,Dy,phi,Z} and x(n+1)=x(n)-grad(f(x(n)))
    // f(x)=sum{i=1 to N}[(Dx-Z(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))^2] + sum{i=1 to N}[(Dy-Z(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))^2]
    // grad(f(x))={df/dDx,df/dDy,df/dphi,df/dZ}
    
    //initial guess
    Dx=0.001;Dy=0.001;phi=0;Z=1.5; 

    // Initial error
    e=0;
    for(size_t i = 0; i < N; i++){
        e =e+(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
    }

    // Iterate x_vect={Dx,Dy,phi,Z} until error<0.01
    count=0; 	//no of iterations for error to converge
    float grad_sum=10;	// sum of squares of gradients    
    
    while((e>=0.0001)&&(count<100)&&(grad_sum>=0.0001)){
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
        
	float dDx=df_dDx(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
	float dDy=df_dDy(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
	float dphi=df_dphi(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
	float dZ=df_dZ(Dx_o,Dy_o,phi_o,Z_o,A,B,N);
	grad_sum=dDx*dDx + dDy*dDy + dphi*dphi + dZ*dZ;    

	// Find error
	e=0;
	for(size_t i = 0; i < N; i++){
	    e = e + (Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0]))+(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]))*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1]));
	}
    }

    if (abs(phi)<=min_phi) {
    	phi=0; 		// to remove accumulation of small 0 error
    }
    if (abs(phi)>=max_phi) {	// to remove impractical values
    	if (phi>0) phi=max_phi; 	
    	else phi=-max_phi; 	
    }
    if (abs(phi)>=max_phi || N<=min_N) phi_status=false; 	// min 10 features change phi_status flag
    else phi_status=true;  
}


void MonoVisualOdometry::calcPoseVector() {
   switch(method){
   	case 1: rotationActualTranslation();
   	break;
   	case 2: rotationScaledTranslation();
   	break;   	
   	case 3: rotationScaledTranslation_reg(); 
  	break;
   	case 4: estimateTransformMatrix();   
   	break;   	
   }
}

void MonoVisualOdometry::updateMotion(){
    // net pose calculation (wrt starting pose) 
    Rcos=Dx*cos(phi)+Dy*sin(phi);
    Rsin=Dx*sin(phi)-Dy*cos(phi);
    rel_Dx=Rcos*cos(net_phi)-Rsin*sin(net_phi);		//relative Dx wrt to previous frame
    rel_Dy=Rcos*sin(net_phi)+Rsin*cos(net_phi);		//relative Dy wrt to previous frame
    rel_phi=phi;					//relative phi wrt to previous frame
    net_Dx=net_Dx+Rcos*cos(net_phi)-Rsin*sin(net_phi); //net(absolute) camera translation in x-direction wrt to starting pose
    net_Dy=net_Dy+Rcos*sin(net_phi)+Rsin*cos(net_phi); //net(absolute) camera translation in y-direction wrt to starting pose	
    net_phi=net_phi+phi; 			       //net(absolute) heading angle (anti-clk +ve)
    Zsum=Zsum+Z;					   
    net_Z1=Zsum/(nframes-1);			       //average estimated_1 value of depth of ground from camera
    if(nframes==2) net_Z2=Z;
    else net_Z2=(net_Z2+Z)/2;			       //average estimated_2 value of depth of ground from camera
} 

void MonoVisualOdometry::run() {
    // clear old vectors
    keypoints1.clear();
    keypoints2.clear();
    keypoints1_2f.clear();    
    keypoints2_2f.clear();    
    matches.clear();
    good_matches.clear();
    fmatches.clear();

    // start the timer
    time=clock();    
    
    // convert to grayscale
    cvtColor(img1,img1,CV_BGR2GRAY);
    cvtColor(img2,img2,CV_BGR2GRAY);  
    
  if(opticalFlow){
    //calculate matched feature points optical flow
    calcOpticalFlow();
  }
  else {
    // find keypoints
    findKeypoints();  
  
    // find descriptors
    findDescriptor();    
  
    // find matches
    findMatches();
  
    // find good_matches
    findGoodMatches();
  }
    // calc normalised 3D coordinates
    calcNormCoordinates();
  
    // calc pose vector x_vect={Dx,Dy,phi,Z}
    calcPoseVector();
  
    // update motion history
    updateMotion();
    
    time=clock()-time;
    run_time=((float)time)/CLOCKS_PER_SEC;   //time for single run

    // drawing the keypoints in two imgs
    const Scalar& color=Scalar(255,255,0); //BGR
    int flags=DrawMatchesFlags::DEFAULT;
    namedWindow("keypoints1", 1);
    Mat img_key1;
    drawKeypoints(img1, keypoints1,img_key1,color,flags);
    imshow("keypoints1", img_key1);
    waitKey(1);
    
    namedWindow("keypoints2", 1);
    Mat img_key2;
    drawKeypoints(img2, keypoints2,img_key2,color,flags);
    imshow("keypoints2", img_key2);
    waitKey(1);    
    
    namedWindow("mask", 1);
    imshow("mask", mask);
    waitKey(1);    

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
    position.x_rel=rel_Dx;
    position.y_rel=rel_Dy;
    position.heading_rel=rel_phi;
    position.rot=rot;
    position.x_scaled=tx;
    position.y_scaled=ty;
    position.error=e;
    position.head_status=phi_status;
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

float MonoVisualOdometry::d2f_d2phi(float Dx,float Dy, float phi, float Z, float **A, float **B, int N) {
  float sum=0;
  for(int i=0;i<N;i++)
  {
    sum=sum + 2*((-Z)*(-A[i][0]*sin(phi)-A[i][1]*cos(phi))) * ((-Z)*(-A[i][0]*sin(phi)-A[i][1]*cos(phi))) + 2*(Dx-Z*(A[i][0]*cos(phi)-A[i][1]*sin(phi)-B[i][0])) * ((-Z)*(-A[i][0]*cos(phi)+A[i][1]*sin(phi))) + 2*((-Z)*(A[i][0]*cos(phi)-A[i][1]*sin(phi))) * ((-Z)*(A[i][0]*cos(phi)-A[i][1]*sin(phi))) + 2*(Dy-Z*(A[i][0]*sin(phi)+A[i][1]*cos(phi)-B[i][1])) * ((-Z)*(-A[i][0]*sin(phi)-A[i][1]*cos(phi)));
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

