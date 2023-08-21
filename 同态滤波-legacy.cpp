#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
// PPM图像格式的简单读取函数
bool readPPM(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data) {
    std::ifstream input_file("C:/Users/office/Desktop/同态滤波/349293-20181010164333331-2001442514.ppm"); 
    if (!input_file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string magic_number;
    input_file >> magic_number;
    if (magic_number != "P6") {
        std::cout << "Invalid PPM format: " << filename << std::endl;
        return false;
    }

    input_file >> width >> height;
    int max_value;
    input_file >> max_value;
    input_file.ignore(1);

    data.resize(width * height * 3);
    input_file.read(reinterpret_cast<char*>(data.data()), width * height * 3);

    return true;
}

// PPM图像格式的简单写入函数
bool writePPM(const std::string& filename, int width, int height, const std::vector<unsigned char>& data) {
    std::ofstream ofs("C:/Users/office/Desktop/同态滤波");
    if (!ofs.is_open()) {
        return false;
    }

    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    ofs.close();

    return true;
}

// 将彩色图像转换为灰度图像
void convertToGrayscale(const std::vector<unsigned char>& input, int width, int height, std::vector<double>& output) {
    output.resize(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width * 3 + j * 3;
            double gray_value = 0.299 * input[index] + 0.587 * input[index + 1] + 0.114 * input[index + 2];
            output[i * width + j] = gray_value;
        }
    }
}

// 对图像中的每个像素应用对数变换
void applyLogTransform(std::vector<double>& input, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            input[i * width + j] = std::log(input[i * width + j] + 1.0);
        }
    }
}

// 使用傅里叶变换计算图像的二维离散傅里叶变换
void computeDFT(const std::vector<double>& input, int width, int height, std::vector<std::vector<std::complex<double> > >& output) {
    output.resize(height);
    for (int i = 0; i < height; ++i) {
        output[i].resize(width);
        for (int j = 0; j < width; ++j) {
            std::complex<double> sum(0.0, 0.0);
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    double angle = 2.0 * M_PI * (i * k / static_cast<double>(height) + j * l / static_cast<double>(width));
                    std::complex<double> expterm(std::cos(angle), -std::sin(angle));
                    sum += input[k * width + l] * expterm;
                }
            }
            output[i][j] = sum;
        }
    }
}

// 使用傅里叶变换计算图像的二维傅里叶逆变换
void computeIDFT(const std::vector<std::vector<std::complex<double> > >& input, int width, int height, std::vector<double>& output) {
    output.resize(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::complex<double> sum(0.0, 0.0);
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    double angle = 2.0 * M_PI * (i * k / static_cast<double>(height) + j * l/ static_cast<double>(width));
                    std::complex<double> expterm(std::cos(angle), std::sin(angle));
                    sum += input[k][l] * expterm;
                }
            }
            output[i * width + j] = std::real(sum) / (width * height);
        }
    }
}

int main() {
    // 读取输入图像
    int width, height;
    std::vector<unsigned char> input_data;
    if (!readPPM("input.ppm", width, height, input_data)) {
        return 1;
	}
	std::cout<<"1";
    // 将彩色图像转换为灰度图像
    std::vector<double> grayscale_data;
    convertToGrayscale(input_data, width, height, grayscale_data);
	std::cout<<"2";
    // 应用对数变换
    applyLogTransform(grayscale_data, width, height);
	std::cout<<"3";
    // 计算离散傅里叶变换
    std::vector<std::vector<std::complex<double> > > dft_data;
    computeDFT(grayscale_data, width, height, dft_data);
	std::cout<<"4";
    // 频谱平移
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int shift_i = (i + height / 2) % height;
            int shift_j = (j + width / 2) % width;
            std::swap(dft_data[i][j], dft_data[shift_i][shift_j]);
        }
    }
	std::cout<<"5";
    // 计算逆离散傅里叶变换
    std::vector<double> idft_data;
    computeIDFT(dft_data, width, height, idft_data);
	std::cout<<"6";
    // 对逆变换结果应用指数变换
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            idft_data[i * width + j] = std::exp(idft_data[i * width + j]) - 1.0;
        }
    }
	std::cout<<"7";
    // 将结果图像写入文件
    std::vector<unsigned char> output_data(width * height * 3);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width * 3 + j * 3;
            output_data[index] = idft_data[i * width + j];
            output_data[index + 1] = idft_data[i * width + j];
            output_data[index + 2] = idft_data[i * width + j];
        }
    }
    writePPM("output.ppm", width, height, output_data);
	std::cout<<"8";
    return 0;
}
