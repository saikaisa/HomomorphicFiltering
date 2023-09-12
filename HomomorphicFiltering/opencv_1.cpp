#include <opencv2/opencv.hpp>
using namespace std;

using namespace cv;
// Ӧ��̬ͬ�˲����ĺ���
Mat HomoFilter(cv::Mat src) {
    // ��ԭʼͼ��ת��Ϊ˫����
    src.convertTo(src, CV_64FC1);

    // ��鲢����ͼ���СΪż��
    int rows = src.rows;
    int cols = src.cols;
    int m = rows % 2 == 1 ? rows + 1 : rows;
    int n = cols % 2 == 1 ? cols + 1 : cols;
    copyMakeBorder(src, src, 0, m - rows, 0, n - cols, BORDER_CONSTANT, Scalar::all(0));
    rows = src.rows;
    cols = src.cols;

    // �������ڴ洢�˲���ͼ���Ŀ�����
    Mat dst(rows, cols, CV_64FC1);

    // 1.�����任
    for (int i = 0; i < rows; i++) {
        double* srcdata = src.ptr<double>(i);
        double* logdata = src.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            logdata[j] = log(srcdata[j] + 0.0001);
        }
    }
    // 2.������ɢ���ұ任��DCT)
    Mat mat_dct = Mat::zeros(rows, cols, CV_64FC1);
    dct(src, mat_dct);

    // 3. ��˹̬ͬ�˲���
    Mat H_u_v;
    double gammaH = 2.0;
    double gammaL = 0.2;
    double C = 3;
    double d0 = (src.rows / 2) * (src.rows / 2) + (src.cols / 2) * (src.cols / 2);
    double d2 = 0;
    H_u_v = Mat::zeros(rows, cols, CV_64FC1);
    for (int i = 0; i < rows; i++) {
        double* dataH_u_v = H_u_v.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            d2 = pow(i, 2.0) + pow(j, 2.0);
            dataH_u_v[j] = (gammaH - gammaL) * (1 - exp(-C * d2 / d0)) + gammaL;
        }
    }
    H_u_v.ptr<double>(0)[0] = 1.0;
    mat_dct = mat_dct.mul(H_u_v);

    // 4.��������ɢ���ұ任��IDCT��
    idct(mat_dct, dst);

    // ָ���任
    for (int i = 0; i < rows; i++) {
        double* srcdata = dst.ptr<double>(i);
        double* dstdata = dst.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            dstdata[j] = exp(srcdata[j]);
        }
    }
     // ת��Ϊ8λ�޷�������ͼ��
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

int main() {
    // ��ȡԭʼͼ��
    Mat src = cv::imread("349293-20181010164333331-2001442514.jpg");
    imshow("origin", src);

    int originrows = src.rows;
    int origincols = src.cols;
    Mat dst(src.rows, src.cols, CV_8UC3);

    // ת��ΪYUV��ɫ�ռ�
    cvtColor(src, src, COLOR_BGR2YUV);
    vector<Mat> yuv;
    split(src, yuv);
    Mat nowY = yuv[0];

    // Ӧ��̬ͬ�˲���
    Mat newY = HomoFilter(nowY);

    // ��������Ƶ���ʱ����
    Mat tempY(originrows, origincols, CV_8UC1);
    for (int i = 0; i < originrows; i++) {
        for (int j = 0; j < origincols; j++) {
            tempY.at<uchar>(i, j) = newY.at<uchar>(i, j);
        }
    }
    yuv[0] = tempY;

    // �ϲ�ͨ����ת����BGR��ɫ�ռ�
    merge(yuv, dst);
    cvtColor(dst, dst, COLOR_YUV2BGR);

    imshow("result", dst);
    waitKey(0);

    return 0;
}