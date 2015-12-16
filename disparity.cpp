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
#include "cstring"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "libelas/elas.h"
#include "libelas/image.h"
#include "libelas/process.h"

using namespace std;
using namespace cv;

string get_sequence(int n) {
	if (n==0) return "000000";
	else if (n/10 == 0) return "00000"+to_string(n);
	else if (n/100 == 0) return "0000"+to_string(n);
	else if (n/1000 == 0) return "000"+to_string(n);
	return NULL;
}

int main(int argc, char** argv) {
	if (argc< 3) {
		cout << "Enter path to data.. ./stereo <path> <numFiles>\n";
		return -1;
	}
	
	if (argv[1][strlen(argv[1])-1] == '/') {
		argv[1][strlen(argv[1])-1] = '\0';
	}

	string path = string(argv[1]);
	int SEQ_MAX = atoi(argv[2]);

	Mat left_frame, right_frame, disp8U, disp16S;
	Mat left_frame_rgb, right_frame_rgb, left_frame_scaled, right_frame_scaled, disp8U_scaled;
	double max_, min_ ;
	
	string create_dir_command = "mkdir disparity";
	system(create_dir_command.c_str());
	cout << "generating disparity maps..\n";

	for (int i = 0; i<= SEQ_MAX; i++) {
		left_frame_rgb = imread("./"+path+"/I1_"+get_sequence(i)+".png");
		right_frame_rgb = imread("./"+path+"/I2_"+get_sequence(i)+".png");
		
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
  		param.postprocess_only_left = true;
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
        imwrite("./disparity/"+path+"I1_"+get_sequence(i)+".jpg", disp);
        cout << "\033[Fdisparity generated : "+ to_string(((float)i*100)/SEQ_MAX) + "%\n";
    }
	return 0;
}