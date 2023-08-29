#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // ��ȡͼ��
    Mat img = imread("input.jpg", IMREAD_GRAYSCALE);

    // ת��Ϊ����������
    img.convertTo(img, CV_32F);

    // �����任
    log(img + 1, img);

    // ����Ҷ�任
    dft(img, img, DFT_COMPLEX_OUTPUT);

    // �����˲���
    Mat filter(img.size(), CV_32FC2);
    float gamma_l = 0.5;
    float gamma_h = 1.5;
    float c = 1.0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float d = sqrt(pow(i - img.rows / 2, 2) + pow(j - img.cols / 2, 2));
            filter.at<Vec2f>(i, j)[0] = (gamma_h - gamma_l) * (1 - exp(-c * pow(d, 2) / pow(img.rows / 2, 2))) + gamma_l;
            filter.at<Vec2f>(i, j)[1] = (gamma_h - gamma_l) * (1 - exp(-c * pow(d, 2) / pow(img.rows / 2, 2))) + gamma_l;
        }
    }

    // �˲�
    mulSpectrums(img, filter, img, 0);

    // ����Ҷ���任
    dft(img, img, DFT_INVERSE | DFT_SCALE);

    // �������任
    exp(img, img);

    // ת��Ϊ 8 λ�޷�����������
    img.convertTo(img, CV_8U);

    // ��ʾͼ��
    imshow("output", img);
    waitKey(0);

    return 0;
}