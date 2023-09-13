#include <opencv2/opencv.hpp>
using namespace std;

using namespace cv;
// 应用同态滤波器的函数
Mat HomoFilter(cv::Mat src, double gammaH, double gammaL, double C) {
    // 将原始图像转换为双精度
    src.convertTo(src, CV_64FC1);

    // 检查并调整图像大小为偶数
    int rows = src.rows;
    int cols = src.cols;
    int m = rows % 2 == 1 ? rows + 1 : rows;
    int n = cols % 2 == 1 ? cols + 1 : cols;
    copyMakeBorder(src, src, 0, m - rows, 0, n - cols, BORDER_CONSTANT, Scalar::all(0));
    rows = src.rows;
    cols = src.cols;

    // 创建用于存储滤波后图像的目标矩阵
    Mat dst(rows, cols, CV_64FC1);

    // 1.对数变换
    for (int i = 0; i < rows; i++) {
        double* srcdata = src.ptr<double>(i);
        double* logdata = src.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            logdata[j] = log(srcdata[j] + 1);
        }
    }
    // 2.进行离散余弦变换（DCT)
    Mat mat_dct = Mat::zeros(rows, cols, CV_64FC1);
    dct(src, mat_dct);

    // 3. 高斯同态滤波器
    //double d0 = max(cols, rows);  // 处理彩色图像
    double d0 = 100;    // 处理黑白图像
    Mat H_u_v = Mat::zeros(rows, cols, CV_64FC1);
    double centerX = cols / 2.0;
    double centerY = rows / 2.0;
    for (int y = 0; y < rows; y++) {
        double* rowPtr = H_u_v.ptr<double>(y);   // 指向行的指针
        for (int x = 0; x < cols; x++) {
            //double distance = (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY); // 处理彩色图像时效果较好
            double distance = x * x + y * y;    // 处理黑白图像时效果较好
            rowPtr[x] = (gammaH - gammaL) * (1 - exp(-C * distance / (2 * d0 * d0))) + gammaL;
        }
    }
    H_u_v.ptr<double>(0)[0] = 1.0;  // 将高斯同态滤波器的响应函数在频率平面的原点设置为1.0，保持图像的整体亮度不受滤波器的影响
    mat_dct = mat_dct.mul(H_u_v);   // 矩阵对应像素相乘

    // 4.进行逆离散余弦变换（IDCT）
    idct(mat_dct, dst);

    // 指数变换
    for (int i = 0; i < rows; i++) {
        double* srcdata = dst.ptr<double>(i);
        double* dstdata = dst.ptr<double>(i);
        for (int j = 0; j < cols; j++) {
            dstdata[j] = exp(srcdata[j]) - 1;
        }
    }
    // 转换为8位无符号整型图像
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

int main() {
    // 读取原始图像
    Mat src = imread("black_and_white_image.ppm");
    imshow("origin", src);

    int originrows = src.rows;
    int origincols = src.cols;
    Mat dst(src.rows, src.cols, CV_8UC3);

    // 转换为YUV颜色空间
    cvtColor(src, src, COLOR_BGR2YUV);  // 原RGB，现YUV
    vector<Mat> yuv;    // 用于存储 3 个单通道图像（分别为Y、U、V通道）
    split(src, yuv);    // 将一个三通道图像src 分离为3个单通道图像yuv[0],yuv[1],yuv[2]
    Mat nowY = yuv[0];  // nowY存储的是 Y 通道图像

    double gammaH = 2.0;
    double gammaL = 0.2;
    double C = 2; // 越高越暗
    // 应用同态滤波器
    Mat newY = HomoFilter(nowY, gammaH, gammaL, C);

    // 创建一个新的临时Mat对象tempY，并将newY的像素值复制给tempY
    Mat tempY(originrows, origincols, CV_8UC1);
    for (int i = 0; i < originrows; i++) {
        for (int j = 0; j < origincols; j++) {
            tempY.at<uchar>(i, j) = newY.at<uchar>(i, j);
        }
    }
    yuv[0] = tempY;
   
    // 给 Y 通道归一化
    Mat normalizedY;
    normalize(yuv[0], yuv[0], 0, 255, NORM_MINMAX, CV_8UC1);

    // 合并通道并转换回BGR颜色空间
    merge(yuv, dst);    // 与split相反，merge是将3个单通道图像组合成1个三通道图像
    cvtColor(dst, dst, COLOR_YUV2BGR);  // 原YUV，现RGB

    //// 给 BGR 图像归一化，效果很差
    //Mat normalizedDst;
    //normalize(dst, normalizedDst, 0, 255, NORM_MINMAX, CV_8UC3);

    imshow("result", dst);  // 显示成品
    waitKey(0);

    return 0;
}