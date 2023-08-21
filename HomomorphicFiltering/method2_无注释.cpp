#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <algorithm>
// PPM图像格式的简单读取函数
bool readPPM(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        std::cout << "无法打开文件:  " << filename << std::endl;
        return false;
    }
    std::string magic_number;
    input_file >> magic_number;
    if (magic_number != "P6") {
        std::cout << "文件不是 PPM 格式: " << filename << std::endl;
        return false;
    }
    input_file >> width >> height;
    int max_value;
    input_file >> max_value;
    input_file.ignore(1);
    data.resize(width * height * 3);
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
void convertToYUV(const std::vector<unsigned char>& input, int width, int height, std::vector<double>& yuv_image) {
    yuv_image.resize(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        float r = input[i * 3];
        float g = input[i * 3 + 1];
        float b = input[i * 3 + 2];
        double y = 0.299 * r + 0.587 * g + 0.114 * b;
        double u = -0.14713 * r - 0.28886 * g + 0.436 * b;
        double v = 0.615 * r - 0.51498 * g - 0.10001 * b;
        yuv_image[i * 3] = y;
        yuv_image[i * 3 + 1] = u;
        yuv_image[i * 3 + 2] = v;
    }
}
// 对YUV图像的Y通道进行对数变换
void applyLogTransform(std::vector<double>& input, int width, int height, std::vector<double>& output) {
    output.resize(width * height);
    // 对 Y 通道应用对数变换。std::log1p(x) = log(1 + x)
    for (int i = 0; i < width * height; i++) {
        output[i] = std::log1p(input[i * 3]);
    }
}
// 使用FFTW库计算Y通道的二维离散傅里叶变换
void computeFFT(const std::vector<double>& input, int width, int height, std::vector<std::complex<double> >& output) {
    output.resize(width * height);
    fftw_complex* input_data = fftw_alloc_complex(width * height);
    fftw_complex* output_data = reinterpret_cast<fftw_complex*>(output.data());
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input_data[index][0] = input[index];
            input_data[index][1] = 0.0;
        }
    }
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
    double min_val = yuv_image[0];
    double max_val = yuv_image[0];
    for (int i = 0; i < width * height; i++) {
        double y = yuv_image[i * 3];
        min_val = std::min(min_val, y);
        max_val = std::max(max_val, y);
    }
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
    int width, height;
    std::vector<unsigned char> image_data;
    std::string input_file = "F:\\桌面\\HomomorphicFiltering\\x64\\Debug\\input_image.ppm";
    std::string output_file = "F:\\桌面\\HomomorphicFiltering\\x64\\Debug\\output_image.ppm";
    double D0 = 50.0;
    double gamma_l = 0.25;
    double gamma_h = 1.5;
    double c = 1.0;
    if (!readPPM(input_file, width, height, image_data)) {
        std::cout << "Step  1: 读取PPM图像\t\t\t\t\t失败" << std::endl;
        return 1;
    }
    std::vector<double> yuv_image;
    convertToYUV(image_data, width, height, yuv_image);
    std::vector<double> log_y_channel;
    applyLogTransform(yuv_image, width, height, log_y_channel);
    std::vector<std::complex<double> > fft_output;
    computeFFT(log_y_channel, width, height, fft_output);
    applyFilter(fft_output, width, height, D0, gamma_l, gamma_h, c);
    std::vector<double> ifft_output;
    computeInverseFFT(fft_output, width, height, ifft_output);
    applyInverseLogTransform(ifft_output, width, height, yuv_image);
    normalizeYChannel(yuv_image, width, height);
    std::vector<unsigned char> output_image;
    convertToRGB(yuv_image, width, height, output_image);
    if (!writePPM(output_file, width, height, output_image)) {
        std::cout << "Step 10: 将处理后的灰度图像保存为PPM格式\t\t\t失败" << std::endl;
        return 1;
    }

    return 0;
}
