#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // 读取图像
    Mat img = imread("input.jpg", IMREAD_GRAYSCALE);

    // 转换为浮点数类型
    img.convertTo(img, CV_32F);

    // 对数变换
    log(img + 1, img);

    // 傅里叶变换
    dft(img, img, DFT_COMPLEX_OUTPUT);

    // 构建滤波器
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

    // 滤波
    mulSpectrums(img, filter, img, 0);

    // 傅里叶反变换
    dft(img, img, DFT_INVERSE | DFT_SCALE);

    // 反对数变换
    exp(img, img);

    // 转换为 8 位无符号整数类型
    img.convertTo(img, CV_8U);

    // 显示图像
    imshow("output", img);
    waitKey(0);

    return 0;
}