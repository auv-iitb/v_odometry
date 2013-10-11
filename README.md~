Monocular Visual Odometry

1) odometry.cpp - calculates Dx,Dy,phi,Z {x-translation,y-translation,angle,depth}
It also has various options to choose for feature detection, extraction, mathcing, finding good matches, and using different solving algorithms.

Assumption: The image sequensces are taken such that the camera almost moves parallel to ground, so that the Z coordinates of all the points in each image sequence is nearly tha same.

2) only_rotation.cpp - calculates only the rotation angle assuming no translation

This is built from odometry.cpp as the base and simply iterating over only rotation angle assuming no translation and using the fact that depth has no effect on pure rotation calculations.

3) FAST_GD.cpp - uses FAST for feature detection, SURF for feature extraction, and bruteforce matcher for matching and gradient descent for minimization problem.

4) FAST_GD2.cpp - uses similar approach as FAST_GD.cpp but uses different solving approach using gradient descent.

5) 1.png & 2.png are test files actually taken from stereo-camera which can be used to test the displacement outputs. {x-displ should be some finite value and y-displ should be approx 0 and phi~0}

6) FAST_matrix.cpp - uses c++ matrix library for calculating matrix inverse.

7) vid_odometry.cpp - monocular visual odometry from live video capture

8) visual_odometry_method.txt - method used for odometry is explained

9) MonoVisualOdometry - Monocular Vision Odometry class. Calculates the net pose at any instant.
