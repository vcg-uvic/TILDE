
#include 	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<string>
#include 	<iomanip>

// #include	<cv.h>
// #include	<highgui.h>
#include	<opencv2/opencv.hpp>
#include	<opencv2/features2d/features2d.hpp>
#include	<opencv2/opencv_modules.hpp>
#include	<opencv2/nonfree/features2d.hpp>
#include	<opencv2/nonfree/nonfree.hpp>
// #ifndef WIN32
// #include	<opencv2/core/utility.hpp>
// #include	<opencv2/imgcodecs.hpp>
// #include	<opencv2/highgui.hpp>
// #else
#include	<opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

int
main(int argc, char *argv[]) {

	if (argc != 4){
		cout << "Usage: <Method> <InputImg> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

	// Load the image
	Mat imgColor = imread(argv[2]);
	Mat imgGray;
	cvtColor(imgColor, imgGray, CV_RGB2GRAY );

	initModule_nonfree();

    Ptr<FeatureDetector>		myDetector;

	myDetector = FeatureDetector::create(argv[1]);
	if (!myDetector)
		printf("OpenCV was built without keypoint\n" );
	// myDetector->set("contrastThreshold", 0.04);
	// myDetector->set("edgeThreshold", 10);
	// myDetector->set("sigma", 1.6);

	// Detect keypoints
	vector<KeyPoint> keypts;
	myDetector->detect(imgGray, keypts);

	string score_name(argv[3]); score_name.append(".score");
	ofstream ofs_keypoints;
	ofstream ofs_score;
	ofs_keypoints.open(argv[3], std::ofstream::trunc);
	ofs_score.open(score_name, std::ofstream::trunc);
	// Save keypoints
	ofs_keypoints << 1 << endl;
	ofs_keypoints << keypts.size() << endl;
	for(int i=0; i < keypts.size(); ++i){
		ofs_keypoints << std::setprecision(10) << keypts[i].pt.x << " ";
		ofs_keypoints << std::setprecision(10) << keypts[i].pt.y << " ";
		ofs_keypoints << std::setprecision(10) << 1.0 / (0.5*keypts[i].size*0.5*keypts[i].size) << " "; // 0.5 since
																										// opencv uses
																										// diameter
		ofs_keypoints << 0 << " ";
		ofs_keypoints << std::setprecision(10) << 1.0 / (0.5*keypts[i].size*0.5*keypts[i].size) ;
		ofs_keypoints << endl;
		ofs_score << std::setprecision(10) << keypts[i].response << endl;
	}
	ofs_keypoints.close();
	ofs_score.close();

}




