/************************************************************************/
/********************* ���ڲ��� opencv ���Ƿ�װ�ɹ� *********************/
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
		imshow("��ȡ����ͷ", img);
		waitKey(30);
	}
	return 0;
}