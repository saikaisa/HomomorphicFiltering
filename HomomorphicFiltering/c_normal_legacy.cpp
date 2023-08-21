#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <algorithm>
 
// PPM图像格式的简单读取函数
bool readPPM(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data) {
    // 以二进制模式打开文件（图像为一个二进制文件）
    std::ifstream input_file(filename, std::ios::binary);
    // 检测能否打开该文件
    if (!input_file.is_open()) {
        std::cout << "无法打开文件:  " << filename << std::endl; 
        return false; 
    } 
 
    // 检测魔数（PPM的魔数是"P6"，如果不是P6则不是PPM格式的文件） 
    std::string magic_number; 
    input_file >> magic_number;
    if (magic_number != "P6") {
        std::cout << "文件不是 PPM 格式: " << filename << std::endl; 
        return false; 
    } 
 
    // 读取图像的宽高，以及图像的最大颜色值 max_value 
    input_file >> width >> height;
    int max_value;
    input_file >> max_value;
    input_file.ignore(1);
 
    // 调整 image_data 的大小以适应图像大小（每个像素由 3 个颜色分量（红色、绿色和蓝色）组成，所以每个像素占用 3 个字节）
    data.resize(width * height * 3);
 
    /* reinterpret_cast<char*>(data.data()) :
       这是一个类型转换，将 data.data() 返回的 unsigned char 指针转换为 char 指针。
       因为 read() 函数需要一个 char 类型的指针作为参数，所以需要进行类型转换。
 
       这行代码从 PPM 文件中读取 width * height * 3 个字节的像素数据，
       并将这些数据存储在 data 向量中。这样，data 向量就包含了图像的所有像素数据。
    */
    input_file.read(reinterpret_cast<char*>(data.data()), data.size());
 
    return true;
}
 
// PPM图像格式的简单写入函数
bool writePPM(const std::string& filename, int width, int height, const std::vector<unsigned char>& data) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        return false;
    }
 
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    ofs.close();
 
    return true;
}
 
// 将RGB图像转换为YUV图像 (input: image_data, output: yuv_image)
/* 注：
 * 彩色图像使用 R, G, B 存储，范围都在 0~255 之间，所以用 unsigned char 存储比较合适；
 * 但YUV图像的Y通道值由于需要进行后续复杂计算，需要用精确到小数点的形式表示，故用 double 存储。
 */
void convertToYUV(const std::vector<unsigned char>& input, int width, int height, std::vector<double>& yuv_image) {
    yuv_image.resize(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        double r = input[i * 3];
        double g = input[i * 3 + 1];
        double b = input[i * 3 + 2];
        double y = 0.299 * r + 0.587 * g + 0.114 * b;
        double u = -0.14713 * r - 0.28886 * g + 0.436 * b;
        double v = 0.615 * r - 0.51498 * g - 0.10001 * b;
        yuv_image[i * 3] = y;
        yuv_image[i * 3 + 1] = u;
        yuv_image[i * 3 + 2] = v;
    }
}
 
// 对YUV图像的Y通道进行对数变换  (input: yuv_image, output: log_y_channel)
void applyLogTransform(std::vector<double>& input, int width, int height, std::vector<double>& output) {
    output.resize(width * height);
    // 对 Y 通道应用对数变换。std::log1p(x) = log(1 + x)
    for (int i = 0; i < width * height; i++) {
        output[i] = std::log1p(input[i * 3]);
    }
}
 
// 使用FFTW库计算Y通道的二维离散傅里叶变换    (input: log_y_channel, output: fft_output)
void computeFFT(const std::vector<double>& input, int width, int height, std::vector<std::complex<double> >& output) {
    // 将 output 大小调整为图像大小   
    output.resize(width * height);
    // input_data: 对数变换后的灰度图像 的复数表示（傅里叶变换前）
    fftw_complex* input_data = fftw_alloc_complex(width * height);
    // output_data: 输出图像的复数表示（傅里叶变换后）
    fftw_complex* output_data = reinterpret_cast<fftw_complex*>(output.data());
 
    // 将对数变换后的灰度图像 input 复制到 input_data 数组 (double数组 → 复数数组)。但由于输入图像是实数，所以将复数的虚部设置为0。
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input_data[index][0] = input[index];
            input_data[index][1] = 0.0;
        }
    }
 
    // 创建一个正向傅里叶变换计划并执行
    fftw_plan plan = fftw_plan_dft_2d(height, width, input_data, output_data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_free(input_data);
}
 
// 同态滤波器
double homomorphicFilter(double u, double v, double D0, double gamma_l, double gamma_h, double c) {
    double D2 = u * u + v * v;
    double filter_value = (gamma_h - gamma_l) * (1 - std::exp(-c * (D2 / (D0 * D0 * 2)))) + gamma_l;
    return filter_value;
}
 
// 将滤波器系数与傅里叶变换后的实部和虚部数据相乘
void applyFilter(std::vector<std::complex<double>>& fft_output, int width, int height, double D0, double gamma_l, double gamma_h, double c) {
    int center_u = height / 2;
    int center_v = width / 2;
 
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double u = i - center_u;
            double v = j - center_v;
            double filter_value = homomorphicFilter(u, v, D0, gamma_l, gamma_h, c);
 
            int index = i * width + j;
            fft_output[index] *= filter_value;
        }
    }
}
 
// 计算逆傅里叶变换
void computeInverseFFT(const std::vector<std::complex<double>>& input, int width, int height, std::vector<double>& output) {
    output.resize(width * height);
    fftw_complex* output_data = fftw_alloc_complex(width * height);
    fftw_complex* input_data = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(input.data()));
 
    fftw_plan plan = fftw_plan_dft_2d(height, width, input_data, output_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
 
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            output[index] = output_data[index][0] / (width * height);
        }
    }
 
    fftw_free(output_data);
}
 
// 对原YUV图像的Y通道中的每个像素应用逆对数变换
void applyInverseLogTransform(std::vector<double>& input, int width, int height, std::vector<double>& output) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            output[index * 3] = std::exp(input[index]) - 1.0;
        }
    }
}
 
// 归一化Y通道
void normalizeYChannel(std::vector<double>& yuv_image, int width, int height) {
    // 找到Y通道的最大值和最小值
    double min_val = yuv_image[0];
    double max_val = yuv_image[0];
    for (int i = 0; i < width * height; i++) {
        double y = yuv_image[i * 3];
        min_val = std::min(min_val, y);
        max_val = std::max(max_val, y);
    }
 
    // 对Y通道进行归一化
    for (int i = 0; i < width * height; i++) {
        double y = yuv_image[i * 3];
        yuv_image[i * 3] = 255.0 * (y - min_val) / (max_val - min_val);
    }
}
 
// 将YUV图像转换回RGB图像 (input: yuv_image, output: rgb_image)
void convertToRGB(const std::vector<double>& input, int width, int height, std::vector<unsigned char>& output) {
    output.resize(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        double y = input[i * 3];
        double u = input[i * 3 + 1];
        double v = input[i * 3 + 2];
        unsigned char r = std::clamp(static_cast<int>(y + 1.13983 * v), 0, 255);
        unsigned char g = std::clamp(static_cast<int>(y - 0.39465 * u - 0.58060 * v), 0, 255);
        unsigned char b = std::clamp(static_cast<int>(y + 2.03211 * u), 0, 255);
        output[i * 3] = r;
        output[i * 3 + 1] = g;
        output[i * 3 + 2] = b;
    }
}
 
int main() {
    // 图像信息
    int width, height;
    std::vector<unsigned char> image_data;
    std::string input_file = "input_image.ppm";
    std::string output_file = "output_image.ppm";
 
    // 高斯同态滤波器参数
    double D0 = 50.0;       // 截止频率
    double gamma_l = 0.25;  // 低频增益系数
    double gamma_h = 1.5;   // 高频增益系数
    double c = 1.0;         // 高斯滤波器的形状参数
 
    // 1. 读取PPM图像
    if (!readPPM(input_file, width, height, image_data)) {
        std::cout << "Step  1: 读取PPM图像\t\t\t\t\t失败" << std::endl;
        return 1;
    }
    std::cout << "Step  1: 读取PPM图像\t\t\t\t\t成功" << std::endl;
 
    // 2. 将RGB图像转换为YUV图像
    std::vector<double> yuv_image;
    convertToYUV(image_data, width, height, yuv_image);
    std::cout << "Step  2: RGB图像转换为YUV图像\t\t\t\t成功" << std::endl;
 
    // 3. 对YUV图像的Y通道进行对数变换，存储到 log_y_channel 中
    std::vector<double> log_y_channel;
    applyLogTransform(yuv_image, width, height, log_y_channel);
    std::cout << "Step  3: 对YUV图像的Y通道进行对数变换\t\t\t成功" << std::endl;
 
    // 4. 使用FFTW库计算Y通道的二维离散傅里叶变换
    std::vector<std::complex<double>> fft_output;
    computeFFT(log_y_channel, width, height, fft_output);
    std::cout << "Step  4: 使用FFTW库计算Y通道的二维离散傅里叶变换\t成功" << std::endl;
 
    // 5. 将滤波器系数与傅里叶变换后的实部和虚部数据相乘
    applyFilter(fft_output, width, height, D0, gamma_l, gamma_h, c);
    std::cout << "Step  5: 将滤波器系数与傅里叶变换后的实部和虚部数据相乘\t成功" << std::endl;
 
    // 6. 计算逆傅里叶变换
    std::vector<double> ifft_output;
    computeInverseFFT(fft_output, width, height, ifft_output);
    std::cout << "Step  6: 计算逆傅里叶变换\t\t\t\t成功" << std::endl;
 
    // 7. 对 ifft_output 应用逆对数变换，将其拷贝到 yuv_image 的Y通道中
    applyInverseLogTransform(ifft_output, width, height, yuv_image);
    std::cout << "Step  7: 逆对数变换，并将其拷贝到Y通道中\t\t成功" << std::endl;
 
    // 8. 归一化Y通道
    normalizeYChannel(yuv_image, width, height);
    std::cout << "Step  8: 归一化Y通道\t\t\t\t\t成功" << std::endl;
 
    // 9. 将处理后的YUV图像转换回RGB图像
    std::vector<unsigned char> output_image;
    convertToRGB(yuv_image, width, height, output_image);
    std::cout << "Step  9: YUV图像转换回RGB图像\t\t\t\t成功" << std::endl;
 
    // 10. 将处理后的灰度图像保存为PPM格式
    if (!writePPM(output_file, width, height, output_image)) {
        std::cout << "Step 10: 将处理后的灰度图像保存为PPM格式\t\t\t失败" << std::endl;
        return 1;
    }
    std::cout << "Step 10: 将处理后的RGB图像保存为PPM格式\t\t\t成功" << std::endl;
 
    return 0;
}