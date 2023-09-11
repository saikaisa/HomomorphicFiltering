#include <opencv2/opencv.hpp>
using namespace std;

using namespace cv;

Mat HomoFilter(cv::Mat src) {
    src.convertTo(src, CV_64FC1);
    int rows = src.rows;
    int cols = src.cols;
    int m = rows % 2 == 1 ? rows + 1 : rows;
    int n = cols % 2 == 1 ? cols + 1 : cols;
    copyMakeBorder(src, src, 0, m - rows, 0, n - cols, BORDER_CONSTANT, Scalar::all(0));
    rows = src.rows;
    cols = src.cols;
    Mat dst(rows, cols, CV_64FC1);
    // 1. ln
    for (int i = 0; i < rows; i++) {
        double* srcdata = src.ptr<double>(i);
        double* logdata = src.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            logdata[j] = log(srcdata[j] + 0.0001);
        }
    }
    // 2. dct
    Mat mat_dct = Mat::zeros(rows, cols, CV_64FC1);
    dct(src, mat_dct);
    // 3. 高斯同态滤波器
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
    H_u_v.ptr<double>(0)[0] = 1.1;
    mat_dct = mat_dct.mul(H_u_v);
    // 4. idct
    idct(mat_dct, dst);
    // exp
    for (int i = 0; i < rows; i++) {
        double* srcdata = dst.ptr<double>(i);
        double* dstdata = dst.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            dstdata[j] = exp(srcdata[j]);
        }
    }
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

int main() {
    // 读取原始图像
    Mat src = cv::imread("input_image.ppm");
    imshow("origin", src);

    int originrows = src.rows;
    int origincols = src.cols;
    Mat dst(src.rows, src.cols, CV_8UC3);

    // 转换为YUV颜色空间
    cvtColor(src, src, COLOR_BGR2YUV);
    vector<Mat> yuv;
    split(src, yuv);
    Mat nowY = yuv[0];

    // 应用同态滤波器
    Mat newY = HomoFilter(nowY);

    // 将结果复制到临时矩阵
    Mat tempY(originrows, origincols, CV_8UC1);
    for (int i = 0; i < originrows; i++) {
        for (int j = 0; j < origincols; j++) {
            tempY.at<uchar>(i, j) = newY.at<uchar>(i, j);
        }
    }
    yuv[0] = tempY;

    // 合并通道并转换回BGR颜色空间
    merge(yuv, dst);
    cvtColor(dst, dst, COLOR_YUV2BGR);

    imshow("result", dst);
    waitKey(0);

    return 0;
}