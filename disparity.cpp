#include "iostream"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "libelas/elas.h"
#include "libelas/image.h"
#include "libelas/process.h"

using namespace std;
using namespace cv;

#define SEQ_MAX 111

string get_sequence(int n) {
	if (n==0) return "000000";
	else if (n/10 == 0) return "00000"+to_string(n);
	else if (n/100 == 0) return "0000"+to_string(n);
	else if (n/1000 == 0) return "000"+to_string(n);
	return NULL;
}

int main() {
	Mat left_frame, right_frame, disp8U, disp16S;
	Mat left_frame_rgb, right_frame_rgb, left_frame_scaled, right_frame_scaled, disp8U_scaled;
	double max_, min_ ;
	
	string create_dir_command = "mkdir disparity";
	system(create_dir_command.c_str());
	cout << "generating disparity maps..\n";

	for (int i = 0; i<= SEQ_MAX; i++) {
		left_frame_rgb = imread("./2010_03_09_drive_0023/I1_"+get_sequence(i)+".png");
		right_frame_rgb = imread("./2010_03_09_drive_0023/I2_"+get_sequence(i)+".png");
		
		cvtColor(left_frame_rgb, left_frame, CV_BGR2GRAY);
		cvtColor(right_frame_rgb, right_frame, CV_BGR2GRAY);

        // generate disparity image using LIBELAS
        // obtain normalized image
        int bd = 0;
        const cv::Size imsize = left_frame.size();
		const int32_t dims[3] = {imsize.width,imsize.height,imsize.width};
        cv::Mat leftdpf = cv::Mat::zeros(imsize, CV_32F);
		cv::Mat rightdpf = cv::Mat::zeros(imsize, CV_32F);
		Elas::parameters param;
  		param.postprocess_only_left = false;
  		Elas elas(param);
		elas.process(left_frame.data,right_frame.data,leftdpf.ptr<float>(0),rightdpf.ptr<float>(0),dims);
		Mat disp, leftdisp;
		Mat(leftdpf(cv::Rect(bd,0,left_frame.cols,left_frame.rows))).copyTo(disp);
		disp.convertTo(leftdisp,CV_16S,16);
		minMaxLoc(leftdisp, &min_, &max_);
        leftdisp.convertTo(disp, CV_8UC1, 255/(max_ - min_));
        normalize(disp, disp, 0, 255, CV_MINMAX, CV_8UC1);

        // display and save to file
        imshow("disparity", disp);
        waitKey(33);
        imwrite("./disparity/I1_"+get_sequence(i)+".jpg", disp);
        cout << "\033[Fdisparity generated : "+ to_string(((float)i*100)/SEQ_MAX) + "%\n";
    }
	return 0;
}