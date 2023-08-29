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
		imshow("¶ÁÈ¡ÉãÏñÍ·", img);
		waitKey(30);
	}
	return 0;
}