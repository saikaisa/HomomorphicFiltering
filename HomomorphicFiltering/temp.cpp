#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;

// 傅里叶变换得到频谱图和复数域结果
void custom_dft(Mat input_image, Mat& spectrogram, Mat& fft_image) {
    // 1.扩展图像矩阵，为2，3，5的倍数时运算速度快
    int m = getOptimalDFTSize(input_image.rows);
    int n = getOptimalDFTSize(input_image.cols);
    copyMakeBorder(input_image, input_image, 0, m - input_image.rows, 0, n - input_image.cols, BORDER_CONSTANT, Scalar::all(0));

    // 2.创建一个双通道矩阵planes，用来储存复数的实部与虚部
    Mat planes[] = { Mat_<double>(input_image), Mat::zeros(input_image.size(), CV_64F) };

    // 3.从多个单通道数组中创建一个多通道数组:fft_image。函数Merge将几个数组合并为一个多通道阵列，即输出数组的每个元素将是输入数组元素的级联
    merge(planes, 2, fft_image);

    // 4.进行傅立叶变换
    dft(fft_image, fft_image);

    // 5.计算复数的幅值，保存在spectrogram（频谱图）
    split(fft_image, planes);   // 将双通道分为两个单通道，一个表示实部，一个表示虚部
    Mat fft_image_real = planes[0];
    Mat fft_image_imag = planes[1];

    magnitude(planes[0], planes[1], spectrogram);   // 计算复数的幅值，保存在spectrogram（频谱图）

    // 6.前面得到的频谱图数级过大，不好显示，因此转换
    spectrogram += Scalar(1);   // 取对数前将所有的像素都加 1，防止log0
    log(spectrogram, spectrogram);  // 取对数
    normalize(spectrogram, spectrogram, 0, 1, NORM_MINMAX); // 归一化

    // 7.剪切和重分布幅度图像限
    spectrogram = spectrogram(Rect(0, 0, spectrogram.cols & -2, spectrogram.rows & -2));

    // 显示中心化之前的频谱图
    imshow("中心化前的频谱图", spectrogram);

    // 重新排列傅里叶图像中的象限，使原点位于图像中心
    int cx = spectrogram.cols / 2;
    int cy = spectrogram.rows / 2;
    Mat q0(spectrogram, Rect(0, 0, cx, cy));   // 左上区域
    Mat q1(spectrogram, Rect(cx, 0, cx, cy));  // 右上区域
    Mat q2(spectrogram, Rect(0, cy, cx, cy));  // 左下区域
    Mat q3(spectrogram, Rect(cx, cy, cx, cy)); // 右下区域

    // 交换象限中心化
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);  // 左上与右下进行交换
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);  // 右上与左下进行交换

    Mat q00(fft_image_real, Rect(0, 0, cx, cy));    // 左上区域
    Mat q01(fft_image_real, Rect(cx, 0, cx, cy));   // 右上区域
    Mat q02(fft_image_real, Rect(0, cy, cx, cy));   // 左下区域
    Mat q03(fft_image_real, Rect(cx, cy, cx, cy));  // 右下区域
    q00.copyTo(tmp); q03.copyTo(q00); tmp.copyTo(q03);  // 左上与右下进行交换
    q01.copyTo(tmp); q02.copyTo(q01); tmp.copyTo(q02);  // 右上与左下进行交换

    Mat q10(fft_image_imag, Rect(0, 0, cx, cy));    // 左上区域
    Mat q11(fft_image_imag, Rect(cx, 0, cx, cy));   // 右上区域
    Mat q12(fft_image_imag, Rect(0, cy, cx, cy));   // 左下区域
    Mat q13(fft_image_imag, Rect(cx, cy, cx, cy));  // 右下区域
    q10.copyTo(tmp); q13.copyTo(q10); tmp.copyTo(q13);  // 左上与右下进行交换
    q11.copyTo(tmp); q12.copyTo(q11); tmp.copyTo(q12);  // 右上与左下进行交换

    planes[0] = fft_image_real;
    planes[1] = fft_image_imag;
    merge(planes, 2, fft_image);    	//将傅里叶变换结果中心化
}

// 输入参数
void input(double& gammaH, double& gammaL, double& C, double& d0) {
    while (1) {
        cout << "输入 -1 使用默认参数，输入 0 填写参数：";
        int cmd;
        cin >> cmd;
        if (cmd == -1) break;
        else if (cmd == 0) {
            cout << "\ngammaH = ";
            cin >> gammaH;
            cout << "\ngammaL = ";
            cin >> gammaL;
            cout << "\nC = ";
            cin >> C;
            cout << "\nd0 = ";
            cin >> d0;
            cout << endl;
            break;
        }
    }
}

int main() {
    // 读取原始图像
    Mat image_raw = imread("color_image.jpg");
    if (image_raw.empty()) {
        cout << "读取错误" << endl;
        return -1;
    }

    // 输入参数
    double gammaH = 2.0;
    double gammaL = 0.2;
    double C = 0.1;       // 越高越暗
    double d0 = max(image_raw.cols, image_raw.rows);    // 滤波半径
    input(gammaH, gammaL, C, d0);
    imshow("原始图像", image_raw);

    // 转换为YUV颜色空间
    cvtColor(image_raw, image_raw, COLOR_BGR2YUV);  // 原RGB，现YUV
    vector<Mat> image_yuv;    // 用于存储 3 个单通道图像（分别为Y、U、V通道）
    split(image_raw, image_yuv);    // 将一个三通道图像image_raw 分离为3个单通道图像image_yuv[0],image_yuv[1],image_yuv[2]
    Mat image_y = image_yuv[0];  // image_y存储的是 Y 通道图像，只剩下亮度信息，每个像素值将只表示黑白灰度

    // 进行对数变换
    Mat image_ln(image_y.size(), CV_64F);
    for (int i = 0; i < image_y.rows; i++) {
        for (int j = 0; j < image_y.cols; j++) {
            image_ln.at<double>(i, j) = log(image_y.at<uchar>(i, j) + 0.1);
        }
    }

    // 傅里叶变换，image_spectrogram 为可显示的频谱图，image_fft 为傅里叶变换的复数结果
    Mat image_spectrogram, image_fft;
    custom_dft(image_ln, image_spectrogram, image_fft);
    imshow("中心化后的频谱图", image_spectrogram);

    // 进行高斯同态滤波
    Mat planes[] = { Mat_<double>(image_spectrogram), Mat::zeros(image_spectrogram.size(),CV_64F) };
    split(image_fft, planes);   // 分离通道，获取实部虚部
    Mat image_fft_real = planes[0];
    Mat image_fft_imag = planes[1];

    // 频谱图中心坐标
    int center_x = image_fft_real.cols / 2;
    int center_y = image_fft_real.rows / 2;

    // 创建高斯同态滤波矩阵
    Mat h = Mat::zeros(image_fft_real.size(), CV_64F);

    // x 为列，y 为行
    for (int y = 0; y < image_fft_real.rows; y++) {
        double* y_pt = h.ptr<double>(y);   // 指向行的指针
        for (int x = 0; x < image_fft_real.cols; x++) {
            int distance_sq = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
            y_pt[x] = (gammaH - gammaL) * (1 - exp(-C * distance_sq / (d0 * d0))) + gammaL;
        }
    }

    // 矩阵对应像素相乘
    image_fft_real = image_fft_real.mul(h);
    image_fft_imag = image_fft_imag.mul(h);
    planes[0] = image_fft_real;
    planes[1] = image_fft_imag;

    // 高斯同态滤波结果
    Mat image_homo_res;
    merge(planes, 2, image_homo_res);

    // 傅里叶逆变换
    Mat iDft[] = { Mat_<double>(image_spectrogram), Mat::zeros(image_spectrogram.size(), CV_64F) };
    idft(image_homo_res, image_homo_res);   // 傅立叶逆变换，得到实部和虚部
    split(image_homo_res, iDft);    // 分离通道，主要获取 0 通道，即实部

    // 计算复数的幅值（幅值 = sqrt(实部^2 + 虚部^2)），保存在 iDft[0]
    // 得到图像的像素强度，通过幅值可以还原原始图像的亮度信息
    magnitude(iDft[0], iDft[1], iDft[0]);
    
    //// 打印归一化前的实部数值（取中心像素点）
    //std::cout << "归一化前：" << std::endl;
    //std::cout << "实部：" << iDft[0].at<double>(iDft[0].cols / 2, iDft[0].rows / 2) << endl;

    // 归一化处理，否则数值过大，指数变换会溢出(inf)
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);

    //// 打印归一化前的实部数值
    //std::cout << "归一化后：" << std::endl;
    //std::cout << "实部：" << iDft[0].at<double>(iDft[0].cols / 2, iDft[0].rows / 2) << endl;

    // 指数变换
    exp(iDft[0], iDft[0]);
    //// 打印指数变换后的实部数值
    //std::cout << "指数变换后：" << std::endl;
    //std::cout << "实部：" << iDft[0].at<double>(iDft[0].cols / 2, iDft[0].rows / 2) << endl;
	
    // 归一化处理，使图像亮度在 0~255 范围内
    normalize(iDft[0], iDft[0], 0, 255, NORM_MINMAX, CV_8U);

    // 合并通道并转换回BGR颜色空间
    Mat result(image_raw.rows, image_raw.cols, CV_8UC3);
    vector<Mat> yuv_channels = { iDft[0], image_yuv[1], image_yuv[2] };
    merge(yuv_channels, result);    // 与split相反，merge是将3个单通道图像组合成1个三通道图像
    cvtColor(result, result, COLOR_YUV2BGR);  // 原 YUV，现 RGB

    imshow("结果", result);    // 显示处理完的图像
    waitKey(0);

    return 0;
}