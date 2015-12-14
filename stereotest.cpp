#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <pcl/common/common_headers.h>
#include <pcl/io/io.h>
#include <pcl/visualization/pcl_visualizer.h>

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

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
    viewer->addCoordinateSystem ( 1.0 );
    viewer->initCameraParameters ();
    return viewer;
}

void reproject(Mat& disp8U, Mat& img, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ptr, double f, double _cx,
						double _cy, double _tx_inv, double _cx_cx_tx_inv) {
	ptr->clear();
	double px, py, pz, pw;
	unsigned char pb, pg, pr;
	for (int i=0;i<img.rows;i++) {
        uchar* rgb = img.ptr<uchar>(i);
        for (int j=0;j<img.cols;j++) {
            int d = static_cast<unsigned>(disp8U(Rect(j, i, 1, 1)).at<uchar>(0));
            if (d==0) continue;
            double pw = -1.0*(double) (d)*_tx_inv + _cx_cx_tx_inv;
            px = static_cast<double> (j) + _cx;
            py = static_cast<double> (i) + _cy;
            pz = f;

            px/=pw; 
            py/=pw;
            pz/=pw; pz*=-1; //pz inverted

            pb = rgb[3*j];
            pg = rgb[3*j+1];
            pr = rgb[3*j+2];

            pcl::PointXYZRGB point;
            point.x = px;
            point.y = py;
            point.z = pz;

            uint32_t _rgb = ((uint32_t) pr << 16 |
              (uint32_t) pg << 8 | (uint32_t)pb);
            point.rgb = *reinterpret_cast<float*>(&_rgb);
            ptr->push_back(point);
        }
    }
    ptr->width = (int) ptr->points.size();
    ptr->height = 1;
}

void init( Mat& M1, Mat& M2, Mat& D1, Mat& D2,
							Mat& R1, Mat& R2, Mat& P1, Mat& P2,
								Mat& R, Mat& T, Mat& Q) {
	M1 = (Mat_<double>(3, 3) << 8.941981e+02, 0.000000e+00, 6.601406e+02, 0.000000e+00, 8.927151e+02, 2.611004e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);
	M2 = (Mat_<double>(3, 3) << 8.800704e+02, 0.000000e+00, 6.635881e+02, 0.000000e+00, 8.798504e+02, 2.690108e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);

	D1 = (Mat_<double>(1, 5) << -3.695739e-01, 1.726456e-01, -1.281525e-03, 1.188796e-03, -4.284730e-02);
	D2 = (Mat_<double>(1, 5) << -3.753454e-01, 1.843265e-01, -1.307069e-03, 2.190397e-03, -4.989103e-02);

	R1 = (Mat_<double>(3, 3) << 9.999122e-01, 7.482788e-04, -1.323067e-02, -9.196597e-04, 9.999157e-01, -1.295197e-02, 1.321986e-02, 1.296300e-02, 9.998286e-01);
	R2 = (Mat_<double>(3, 3) << 9.996560e-01, -1.479808e-02, -2.165190e-02, 1.507742e-02, 9.998045e-01, 1.279563e-02, 2.145832e-02, -1.311768e-02, 9.996837e-01);

	P1 = (Mat_<double>(3, 4) << 6.452401e+02, 0.000000e+00, 6.359587e+02, 0.000000e+00, 0.000000e+00, 6.452401e+02, 1.941291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00);
	P2 = (Mat_<double>(3, 4) << 6.452401e+02, 0.000000e+00, 6.359587e+02, -3.682632e+02, 0.000000e+00, 6.452401e+02, 1.941291e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00);

	R = (Mat_<double>(3, 3) << 9.998381e-01, 1.610234e-02, 8.033237e-03, -1.588968e-02, 9.995390e-01, -2.586908e-02, -8.446087e-03, 2.573724e-02, 9.996331e-01);
	T = (Mat_<double>(1, 3) << -5.706425e-01, 8.447320e-03, 1.235975e-02);

	Q = (Mat_<double>(4, 4) << 1.00, 0.00, 0.00, -660.1406, 0.00, 1.00, 0.00, -261.1004, 0.00, 0.00, 0.00, 893.4566, 0.00, 0.00, 1.752410659914044, 6.041435750053667);
}

int main() {
	Mat map1x, map1y, map2x, map2y;
	Mat M1, M2, D1, D2, R1, R2, P1, P2, R, T, Q;
	init(M1, M2, D1, D2, R1, R2, P1, P2, R, T, Q);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    	viewer = createVisualizer( point_cloud_ptr );

	Ptr<StereoBM> bm = StereoBM::create(128, 15);
	bm->setPreFilterSize(41);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(-64);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);

	Mat left_frame, right_frame, disp8U, disp16S;
	Mat left_frame_rgb, right_frame_rgb, left_frame_scaled, right_frame_scaled, disp8U_scaled;
	double max_, min_ ;
	
	for (int i = 0; i<= SEQ_MAX; i++) {
		left_frame_rgb = imread("./2010_03_09_drive_0023/I1_"+get_sequence(i)+".png");
		right_frame_rgb = imread("./2010_03_09_drive_0023/I2_"+get_sequence(i)+".png");
		
		cvtColor(left_frame_rgb, left_frame, CV_BGR2GRAY);
		cvtColor(right_frame_rgb, right_frame, CV_BGR2GRAY);

		//resize(left_frame, left_frame, Size(), 0.5, 0.5);
		//resize(right_frame, right_frame, Size(), 0.5, 0.5);
		
        /*bm -> compute (left_frame, right_frame, disp16S);
        minMaxLoc(disp16S, &min_, &max_);
        disp16S.convertTo(disp8U, CV_8UC1, 255/(max_ - min_));
        normalize(disp8U, disp8U, 0, 255, CV_MINMAX, CV_8UC1);*/

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

        // reproject into depth image
        reproject(disp, left_frame_rgb, point_cloud_ptr, Q.at<double>(2, 3), Q.at<double>(0, 3), Q.at<double>(1, 3),
									Q.at<double>(3, 2), Q.at<double>(3, 3));
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr);
		viewer->updatePointCloud<pcl::PointXYZRGB> (point_cloud_ptr, rgb, "reconstruction");
		viewer->spinOnce();

		resize(left_frame, left_frame, Size(), 0.5, 0.5);
		resize(disp8U, disp8U, Size(), 0.5, 0.5);
		resize(disp, disp, Size(), 0.5, 0.5);

		imshow("left frame", left_frame);
		imshow("StereoBM", disp8U);
		imshow("libelas", disp);

		int k = waitKey(0);
		if (k==27) break;		
	}
	destroyAllWindows();
	return 0;
}