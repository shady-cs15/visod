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

// function performs ratiotest
// to determine the best keypoint matches 
// between consecutive poses
void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &good_matches) {
	for (vector<vector<DMatch> >::iterator it = matches.begin(); it!=matches.end(); it++) {
		if (it->size()>1 ) {
			if ((*it)[0].distance/(*it)[1].distance > 0.4f) { //0.5f
				it->clear();
			}
		} else {
			it->clear();
		}
		if (!it->empty()) good_matches.push_back((*it)[0]);
	}
}

double get_dist(double x1, double y1, double z1, double x2, double y2, double z2) {
	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}

// function returns relative scale 
// for translation between two poses,
// scale obtained from stereo disparities
// TODO- return median instead of mean 
// TODO- compute scale absolute distance b/w 2 triangulated points
double getScale(int cur_index, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, Mat& Q) {
	int prev_index = cur_index-1;
	Mat disp1 = imread("./disparity/I1_"+get_sequence(prev_index)+".jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat disp2 = imread("./disparity/I1_"+get_sequence(cur_index)+".jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int vsize = kp1.size();
	
	double temp_sum = 0.;
	int valid_pairs = 0;
	vector<double> scales;
	for (int i=0; i<vsize-1; i++) {
		int j = i + 1;
		int d1i = static_cast<unsigned>(disp1(Rect(kp1[i].pt.x, kp1[i].pt.y, 1, 1)).at<uchar>(0));
		int d2i = static_cast<unsigned>(disp2(Rect(kp2[i].pt.x, kp2[i].pt.y, 1, 1)).at<uchar>(0));
		int d1j = static_cast<unsigned>(disp1(Rect(kp1[j].pt.x, kp1[j].pt.y, 1, 1)).at<uchar>(0));
		int d2j = static_cast<unsigned>(disp2(Rect(kp2[j].pt.x, kp2[j].pt.y, 1, 1)).at<uchar>(0));
		if (d1i!=0 && d2i!=0 && d1j!=0 && d2j!=0) {
			double pw1i = -1.0*(double) (d1i)*Q.at<double>(3, 2) + Q.at<double>(3, 3);
			double pz1i = -Q.at<double>(2, 3);
			double px1i = static_cast<double>(kp1[i].pt.x) + Q.at<double>(0, 3);
			double py1i = static_cast<double>(kp1[i].pt.y) + Q.at<double>(1, 3);
			pz1i/=pw1i; px1i/=pw1i; py1i/=pw1i;
			
			double pw1j = -1.0*(double) (d1j)*Q.at<double>(3, 2) + Q.at<double>(3, 3);
			double pz1j = -Q.at<double>(2, 3);
			double px1j = static_cast<double>(kp1[j].pt.x) + Q.at<double>(0, 3);
			double py1j = static_cast<double>(kp1[j].pt.y) + Q.at<double>(1, 3);
			pz1j/=pw1j; px1j/=pw1j; py1j/=pw1j;

			double pw2i = -1.0*(double) (d2i)*Q.at<double>(3, 2) + Q.at<double>(3, 3);
			double pz2i = -Q.at<double>(2, 3);
			double px2i = static_cast<double>(kp2[i].pt.x) + Q.at<double>(0, 3);
			double py2i = static_cast<double>(kp2[i].pt.y) + Q.at<double>(1, 3);
			pz2i/=pw2i; px2i/=pw2i; py2i/=pw2i;
			
			double pw2j = -1.0*(double) (d2j)*Q.at<double>(3, 2) + Q.at<double>(3, 3);
			double pz2j = -Q.at<double>(2, 3);
			double px2j = static_cast<double>(kp2[j].pt.x) + Q.at<double>(0, 3);
			double py2j = static_cast<double>(kp2[j].pt.y) + Q.at<double>(1, 3);
			pz2j/=pw2j; px2j/=pw2j; py2j/=pw2j;

			//double scale = fabs(pz2i - pz2j) / fabs(pz1i - pz1j);
			double scale = get_dist(px2i, py2i, pz2i, px2j, py2j, pz2j) / get_dist(px1i, py1i, pz1i, px1j, py1j, pz1j);
			if (isinf(scale)||isnan(scale)||scale>10) continue;
			//temp_sum += scale;
			//valid_pairs ++;
			scales.push_back(scale);
		}
	}

	//if (valid_pairs==0) return 0;
	//return temp_sum/valid_pairs;
	if (scales.size()) {
		sort(scales.begin(), scales.end());
		return scales[scales.size()/2];
	}
	else return 0.;
}

int main() {
	int seq_id = 0, scene_id = 0;
	Mat cur_frame, prev_frame, cur_frame_kp, prev_frame_kp;
	cur_frame = imread("./2010_03_09_drive_0023/I1_000000.png");
	
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
	Mat Q;
	Q = (Mat_<double>(4, 4) << 1.00, 0.00, 0.00, -660.1406, 0.00, 1.00, 0.00, -261.1004, 0.00, 0.00, 0.00, 893.4566, 0.00, 0.00, 1.752410659914044, 6.041435750053667);
	Mat top_view = Mat::zeros(1000, 1000, CV_8UC3);

	//added
	Mat R_, t_;
	R_ = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
	t_ = (Mat_<double>(3, 1) << 0., 0., 0.);

	for (int i=1; i<=SEQ_MAX; i++) {
		cur_frame.copyTo(prev_frame);
		cur_frame = imread("./2010_03_09_drive_0023/I1_"+get_sequence(i)+".png");
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

		// compute relative scale
		// from good keypoints
		double scale = getScale(i, good_keypoints_1, good_keypoints_2, Q);
		cout << scale << endl;

		double f = (double)(8.941981e+02 + 8.927151e+02)/2;
		Point2f pp((float)6.601406e+02, (float)2.611004e+02);
		Mat E, R, t, T;
		if (point1.size() >5 && point2.size() > 5) {
			E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0);
			recoverPose(E, point2, point1, R, t, f, pp);
			t*=scale;
			hconcat(R, t, T);
  			vconcat(T, i4, T);
  			pose = T*pose;
  			cout << "pose: " << pose.t() << "\n" ;

  			//added
  			t_ = t_ + (R_*(scale*t));
  			R_ = R*R_;

  			// Euler angles
  			cout << "Rotation: ";
  			double alpha_1 = atan2(R.at<double>(1,2), R.at<double>(2,2)) * 180 / PI;
  			cout << "[" << alpha_1 << ", ";
  			double c = sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(0,1)*R.at<double>(0,1));
  			double alpha_2 = atan2(-R.at<double>(0,2), c) * 180 / PI;
  			cout << alpha_2 << ", ";
  			double s1 = sin(alpha_1);
  			double c1 = cos(alpha_1);
  			double alpha_3 = atan2(s1*R.at<double>(2,0)-c1*R.at<double>(1,0),c1*R.at<double>(1,1)-s1*R.at<double>(2,1)) * 180 /PI; 
  			cout << alpha_3 << "]\n";

			cout << "translation: "<< t.t() << "\n" << endl ;

			// draw in top view
			circle(top_view, Point(420+t_.at<double>(0, 2), (420+t_.at<double>(0, 0))), 3, Scalar(0, 255, 0), -1);
		}

  		resize(img_matches, img_matches, Size(), 0.4, 0.6);
  		imshow("matches", img_matches);
  		imshow("Top view", top_view);
		if (waitKey(0) == 27) break;
	}
	return 0;
}