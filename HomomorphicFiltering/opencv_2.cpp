#include <opencv2/opencv.hpp>
using namespace std;

using namespace cv;
// Ӧ��̬ͬ�˲����ĺ���
Mat HomoFilter(cv::Mat src, double gammaH, double gammaL, double C) {
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
            logdata[j] = log(srcdata[j] + 1);
        }
    }
    // 2.������ɢ���ұ任��DCT)
    Mat mat_dct = Mat::zeros(rows, cols, CV_64FC1);
    dct(src, mat_dct);

    // 3. ��˹̬ͬ�˲���
    //double d0 = max(cols, rows);  // �����ɫͼ��
    double d0 = 100;    // ����ڰ�ͼ��
    Mat H_u_v = Mat::zeros(rows, cols, CV_64FC1);
    double centerX = cols / 2.0;
    double centerY = rows / 2.0;
    for (int y = 0; y < rows; y++) {
        double* rowPtr = H_u_v.ptr<double>(y);   // ָ���е�ָ��
        for (int x = 0; x < cols; x++) {
            //double distance = (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY); // �����ɫͼ��ʱЧ���Ϻ�
            double distance = x * x + y * y;    // ����ڰ�ͼ��ʱЧ���Ϻ�
            rowPtr[x] = (gammaH - gammaL) * (1 - exp(-C * distance / (2 * d0 * d0))) + gammaL;
        }
    }
    H_u_v.ptr<double>(0)[0] = 1.0;  // ����˹̬ͬ�˲�������Ӧ������Ƶ��ƽ���ԭ������Ϊ1.0������ͼ����������Ȳ����˲�����Ӱ��
    mat_dct = mat_dct.mul(H_u_v);   // �����Ӧ�������

    // 4.��������ɢ���ұ任��IDCT��
    idct(mat_dct, dst);

    // ָ���任
    for (int i = 0; i < rows; i++) {
        double* srcdata = dst.ptr<double>(i);
        double* dstdata = dst.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            dstdata[j] = exp(srcdata[j]) - 1;
        }
    }
    // ת��Ϊ8λ�޷�������ͼ��
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

int main() {
    // ��ȡԭʼͼ��
    Mat src = imread("black_and_white_image.ppm");
    imshow("origin", src);

    int originrows = src.rows;
    int origincols = src.cols;
    Mat dst(src.rows, src.cols, CV_8UC3);

    // ת��ΪYUV��ɫ�ռ�
    cvtColor(src, src, COLOR_BGR2YUV);  // ԭRGB����YUV
    vector<Mat> yuv;    // ���ڴ洢 3 ����ͨ��ͼ�񣨷ֱ�ΪY��U��Vͨ����
    split(src, yuv);    // ��һ����ͨ��ͼ��src ����Ϊ3����ͨ��ͼ��yuv[0],yuv[1],yuv[2]
    Mat nowY = yuv[0];  // nowY�洢���� Y ͨ��ͼ��

    double gammaH = 2.0;
    double gammaL = 0.2;
    double C = 2; // Խ��Խ��
    // Ӧ��̬ͬ�˲���
    Mat newY = HomoFilter(nowY, gammaH, gammaL, C);

    // ����һ���µ���ʱMat����tempY������newY������ֵ���Ƹ�tempY
    Mat tempY(originrows, origincols, CV_8UC1);
    for (int i = 0; i < originrows; i++) {
        for (int j = 0; j < origincols; j++) {
            tempY.at<uchar>(i, j) = newY.at<uchar>(i, j);
        }
    }
    yuv[0] = tempY;
   
    // �� Y ͨ����һ��
    Mat normalizedY;
    normalize(yuv[0], yuv[0], 0, 255, NORM_MINMAX, CV_8UC1);

    // �ϲ�ͨ����ת����BGR��ɫ�ռ�
    merge(yuv, dst);    // ��split�෴��merge�ǽ�3����ͨ��ͼ����ϳ�1����ͨ��ͼ��
    cvtColor(dst, dst, COLOR_YUV2BGR);  // ԭYUV����RGB

    //// �� BGR ͼ���һ����Ч���ܲ�
    //Mat normalizedDst;
    //normalize(dst, normalizedDst, 0, 255, NORM_MINMAX, CV_8UC3);

    imshow("result", dst);  // ��ʾ��Ʒ
    waitKey(0);

    return 0;
}