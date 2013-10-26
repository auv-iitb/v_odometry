LDIR=-L/usr/local/lib

LIBS=-lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_nonfree -lopencv_video

proj: test_MonoVisualOdometry.o mono_odometry.o
	g++ -o proj test_MonoVisualOdometry.o mono_odometry.o $(LIBS) $(LDIR) -g
mono_odometry.o: mono_odometry.h mono_odometry.cpp
	g++ -c mono_odometry.cpp $(LIBS) $(LDIR)
test_MonoVisualOdometry.o: mono_odometry.cpp test_MonoVisualOdometry.cpp
	g++ -c test_MonoVisualOdometry.cpp $(LIBS) $(LDIR)
clean:
	rm -f *.o core proj
	
