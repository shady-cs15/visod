/*
The MIT License
Copyright (c) 2015 Satyaki Chakraborty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "iostream"
#include "vector"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "math.h"

#define SEQ_MAX 111
#define PI 3.14159265

using namespace std;
using namespace cv;

string get_sequence(int n) {
	if (n==0) return "000000";
	else if (n/10 == 0) return "00000"+to_string(n);
	else if (n/100 == 0) return "0000"+to_string(n);
	else if (n/1000 == 0) return "000"+to_string(n);
	return NULL;
}

void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &good_matches) {
	for (vector<vector<DMatch> >::iterator it = matches.begin(); it!=matches.end(); it++) {
		if (it->size()>1 ) {
			if ((*it)[0].distance/(*it)[1].distance > 0.5f) {
				it->clear();
			}
		} else {
			it->clear();
		}
		if (!it->empty()) good_matches.push_back((*it)[0]);
	}
}

int main() {
	int seq_id = 0, scene_id = 0;
	Mat cur_frame, prev_frame, cur_frame_kp, prev_frame_kp;
	cur_frame = imread("../2010_03_09_drive_0023/I1_000000.png");
	//resize(cur_frame, cur_frame, Size(), 0.4, 0.6);
	cvtColor(cur_frame, cur_frame, CV_BGR2GRAY);
	vector<KeyPoint> keypoints_1, keypoints_2, good_keypoints_1, good_keypoints_2;
	vector<vector<DMatch> > matches;
	vector<DMatch> good_matches;
	vector<Point2f> point1, point2;
	
	Mat descriptors_1, descriptors_2,  img_matches;
	
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> extractor = ORB::create();
	Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams> (6, 12, 1);
	Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);
	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

	detector->detect(cur_frame, keypoints_2);
	extractor->compute(cur_frame, keypoints_2, descriptors_2);

	Mat_<double> pose = (Mat_<double>(4, 1) << 0.00, 0.00, 0.00, 1.00);
	Mat_<double> i4 = (Mat_<double>(1, 4) << 0.00, 0.00, 0.00, 1.00);

	for (int i=1; i<=SEQ_MAX; i++) {
		cur_frame.copyTo(prev_frame);
		cur_frame = imread("../2010_03_09_drive_0023/I1_"+get_sequence(i)+".png");
		//resize(cur_frame, cur_frame, Size(), 0.4, 0.6);
		cvtColor(cur_frame, cur_frame, CV_BGR2GRAY);
	
		keypoints_1 = keypoints_2;
		descriptors_2.copyTo(descriptors_1);
		detector->detect(cur_frame, keypoints_2);
		extractor->compute(cur_frame, keypoints_2, descriptors_2);
		matches.clear();
		good_matches.clear();
		
		try {
			matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
			ratioTest(matches, good_matches);
		} catch(Exception e) {
			cerr << "seq_id: " << i <<"\n";
		}

		drawMatches( prev_frame, keypoints_1, cur_frame, keypoints_2,
               	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		// TODO track features
		// (if needed)
		

		// Retrieve 2D points from good_matches
		// Compute Essential Matrix, R & T
		good_keypoints_1.clear();
		good_keypoints_2.clear();
		point1.clear();
		point2.clear();
		for ( size_t m = 0; m < good_matches.size(); m++) {
			int i1 = good_matches[m].queryIdx;
			int i2 = good_matches[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints_1.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints_2.size()));
            good_keypoints_1.push_back(keypoints_1[i1]);
            good_keypoints_2.push_back(keypoints_2[i2]);
		}
		KeyPoint::convert(good_keypoints_1, point1, vector<int>());
		KeyPoint::convert(good_keypoints_2, point2, vector<int>());

		double f = (double)(8.941981e+02 + 8.927151e+02)/2;
		Point2f pp((float)6.601406e+02, (float)2.611004e+02);
		Mat E, R, t, T;
		if (point1.size() >5 && point2.size() > 5) {
			E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0);
			recoverPose(E, point2, point1, R, t, f, pp);
			hconcat(R, t, T);
  			vconcat(T, i4, T);
  			pose = T*pose;
  			cout << pose.t() << "\n" ;

  			// Euler angles
  			/*
  			double alpha_1 = atan2(R.at<double>(1,2), R.at<double>(2,2)) * 180 / PI;
  			cout << "[" << alpha_1 << ", ";
  			double c = sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(0,1)*R.at<double>(0,1));
  			double alpha_2 = atan2(-R.at<double>(0,2), c) * 180 / PI;
  			cout << alpha_2 << ", ";
  			double s1 = sin(alpha_1);
  			double c1 = cos(alpha_1);
  			double alpha_3 = atan2(s1*R.at<double>(2,0)-c1*R.at<double>(1,0),c1*R.at<double>(1,1)-s1*R.at<double>(2,1)) * 180 /PI; 
  			cout << alpha_3 << "]\n";*/
  		}

  		resize(img_matches, img_matches, Size(), 0.4, 0.6);
  		imshow("matches", img_matches);
		if (waitKey(0) == 27) break;
	}
	return 0;
}