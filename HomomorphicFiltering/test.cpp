/************************************************************************/
/********************* 用于测试 opencv 库是否安装成功 *********************/
/************************************************************************/
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture(0);
	Mat img;
	while (1)
	{
		capture >> img;
		imshow("读取摄像头", img);
		waitKey(30);
	}
	return 0;
}