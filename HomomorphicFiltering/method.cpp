#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>

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

// 将彩色图像转换为灰度图像 (input: image_data, output: grayscale_image)
/* 注：
 * 彩色图像使用 R, G, B 存储，范围都在 0~255 之间，所以用 unsigned char 存储比较合适；
 * 但灰度值由于需要进行后续复杂计算，需要用精确到小数点的形式表示，故用 double 存储。
 */
void convertToGrayscale(const std::vector<unsigned char>& input, int width, int height, std::vector<double>& output) {
    // 灰度图像每个像素占 1 个字节
    output.resize(width * height);
    for (int i = 0; i < width * height; i++) {
        // 计算每个像素的灰度值
        float r = input[i * 3];
        float g = input[i * 3 + 1];
        float b = input[i * 3 + 2];
        // 将计算出的灰度值存储在 grayscale_image 向量中
        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;  // luma 转换公式
    }
}

// 对图像中的每个像素应用对数变换  (input: grayscale_image)
void applyLogTransform(std::vector<double>& input, int width, int height) {
    // 对每个像素应用对数变换。std::log1p(x) = log(1 + x)
    for (int i = 0; i < width * height; i++) {
        input[i] = std::log1p(input[i]);
    }
}

// 使用FFTW库计算图像的二维离散傅里叶变换    (input: grayscale_image, output: fft_output)
void computeFFT(const std::vector<double>& input, int width, int height, std::vector<std::complex<double> >& output) {
    // 将 output 大小调整为图像大小   
    output.resize(width * height);
    // input_data: 对数变换后的灰度图像 的复数表示（傅里叶变换前）
    fftw_complex* input_data = fftw_alloc_complex(width * height);
    // output_data: 输出图像的复数表示（傅里叶变换后）
    fftw_complex* output_data = reinterpret_cast<fftw_complex*>(output.data());
    // 创建一个正向傅里叶变换计划
    fftw_plan plan = fftw_plan_dft_2d(height, width, input_data, output_data, FFTW_FORWARD, FFTW_ESTIMATE);

    // 将对数变换后的灰度图像 input 复制到 input_data 数组 (double数组 → 复数数组)。但由于输入图像是实数，所以将复数的虚部设置为0。
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input_data[index][0] = input[index];
            input_data[index][1] = 0.0;
        }
    }

    fftw_execute(plan); // 执行计划，进行变换
    fftw_destroy_plan(plan);
    fftw_free(input_data);
}

// 高斯同态滤波器
double gaussianHomomorphicFilter(double u, double v, double D0, double gamma_l, double gamma_h, double c) {
    double D2 = u * u + v * v;
    double filter_value = (gamma_h - gamma_l) * (1 - std::exp(-c * (D2 / (D0 * D0)))) + gamma_l;
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
            double filter_value = gaussianHomomorphicFilter(u, v, D0, gamma_l, gamma_h, c);

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

// 对图像中的每个像素应用逆对数变换
void applyInverseLogTransform(std::vector<double>& input, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input[index] = std::exp(input[index]) - 1.0;
        }
    }
}

// 将灰度图像重新映射到0-255范围
void remapGrayscaleImage(std::vector<double>& input, int width, int height, std::vector<unsigned char>& output) {
    double min_value = input[0];
    double max_value = input[0];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            min_value = std::min(min_value, input[index]);
            max_value = std::max(max_value, input[index]);
        }
    }

    double range = max_value - min_value;
    output.resize(width * height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            output[index] = static_cast<unsigned char>(255.0 * (input[index] - min_value) / range);
        }
    }
}

int main() {
    int width, height;
    std::vector<unsigned char> image_data;
    std::string input_file = "F:\\桌面\\HomomorphicFiltering\\x64\\Debug\\input_image.ppm";
    std::string output_file = "F:\\桌面\\HomomorphicFiltering\\x64\\Debug\\output_image.ppm";

    // 0. 读取PPM图像
    if (!readPPM(input_file, width, height, image_data)) {
        std::cout << "Step 0: 读取PPM图像\t\t\t\t\t失败" << std::endl;
        return 1;
    }
    std::cout << "Step 0: 读取PPM图像\t\t\t\t\t成功" << std::endl;

    // 1. 将彩色图像转换为灰度图像
    std::vector<double> grayscale_image;
    convertToGrayscale(image_data, width, height, grayscale_image);
    std::cout << "Step 1: 将彩色图像转换为灰度图像\t\t\t成功" << std::endl;

    // 2. 对图像中的每个像素应用对数变换
    applyLogTransform(grayscale_image, width, height);
    std::cout << "Step 2: 对图像中的每个像素应用对数变换\t\t\t成功" << std::endl;

    // 3. 使用FFTW库计算图像的二维离散傅里叶变换
    std::vector<std::complex<double> > fft_output;
    computeFFT(grayscale_image, width, height, fft_output);
    std::cout << "Step 3: 使用FFTW库计算图像的二维离散傅里叶变换\t\t成功" << std::endl;

    // 高斯同态滤波器参数
    double D0 = 50.0;
    double gamma_l = 0.25;
    double gamma_h = 1.5;
    double c = 1.0;

    // 4. 将滤波器系数与傅里叶变换后的实部和虚部数据相乘
    applyFilter(fft_output, width, height, D0, gamma_l, gamma_h, c);
    std::cout << "Step 4: 将滤波器系数与傅里叶变换后的实部和虚部数据相乘\t成功" << std::endl;

    // 5. 计算逆傅里叶变换
    std::vector<double> processed_image;
    computeInverseFFT(fft_output, width, height, processed_image);
    std::cout << "Step 5: 计算逆傅里叶变换\t\t\t\t成功" << std::endl;

    // 6. 对图像中的每个像素应用逆对数变换
    applyInverseLogTransform(processed_image, width, height);
    std::cout << "Step 6: 对图像中的每个像素应用逆对数变换\t\t成功" << std::endl;

    // 7. 将灰度图像重新映射到0-255范围
    std::vector<unsigned char> output_image;
    remapGrayscaleImage(processed_image, width, height, output_image);
    std::cout << "Step 7: 将灰度图像重新映射到0-255范围\t\t\t成功" << std::endl;

    // 8. 将处理后的灰度图像保存为PPM格式
    if (!writePPM(output_file, width, height, output_image)) {
        std::cout << "Step 8: 将处理后的灰度图像保存为PPM格式\t\t\t失败" << std::endl;
        return 1;
    }
    std::cout << "Step 8: 将处理后的灰度图像保存为PPM格式\t\t\t成功" << std::endl;

    return 0;
}
