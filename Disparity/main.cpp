#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp" 
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
//using namespace cv::ximgproc;
using namespace std;

int main(int argc, char** argv)
{
	Mat left = imread("left0.jpg", CV_LOAD_IMAGE_GRAYSCALE); //path to image left
	Mat right = imread("right0.jpg", CV_LOAD_IMAGE_GRAYSCALE); //to right
	Mat disparity;
	
	//all disparity param are to be changed
	Ptr<StereoBM> bm = StereoBM::create(-20, 64);
	bm->setMinDisparity(10);
	bm->setBlockSize(19);
	bm->setNumDisparities(240);
	bm->setUniquenessRatio(10);
	bm->setSpeckleWindowSize(10);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(10);
	bm->compute(left, right, disparity);

	//convert to disparity map
	Mat disparity_map;
	double min_limit, max_limit;
	minMaxLoc(disparity, &min_limit, &max_limit);
	disparity.convertTo(disparity_map, CV_8UC1, 255 / (max_limit - min_limit), -255 * min_limit / (max_limit - min_limit));

	//colored map
	Mat imcolor, imcolor_show;
	applyColorMap(disparity_map, imcolor, COLORMAP_JET);
	imshow("result", imcolor);
	imwrite("disparity.png", imcolor);
	
	Mat Q = (Mat_<double>(3, 4) << 1, 0.000000, 0, -1048.710384130478,
		0.000000, 1, 0, -222.1660407781601,
		0.000000, 0.000000, 0.016745, 0); //get Q matrix from camera calibration
	Vec3f point;
	Mat imgDisparity, pointCloud;
	normalize(disparity, imgDisparity, 0, 255, CV_MINMAX, CV_8U);
	resize(imgDisparity, pointCloud, imgDisparity.size());
	reprojectImageTo3D(imgDisparity, pointCloud, Q, true, -1);
	ofstream file;
	int r, c;
	file.open("data.txt"); //file to store points, to be change
	for (r = 0; r < pointCloud.rows; r++) {
		for (int c = 0; c < pointCloud.cols; c++) {
			if (pointCloud.at<Vec3f>(r, c)[2] < 10) {
				file << pointCloud.at<Vec3f>(r, c)[0] << " " << pointCloud.at<Vec3f>(r, c)[1] << " " << pointCloud.at<Vec3f>(r, c)[2] << " " 
					<< static_cast<unsigned>(left.at<uchar>(r, c)) << " " << static_cast<unsigned>(left.at<uchar>(r, c)) << " " << static_cast<unsigned>(left.at<uchar>(r, c)) << endl;
			}
		}
	}
	file.close();
	waitKey(0);
	return 0;
	
}


