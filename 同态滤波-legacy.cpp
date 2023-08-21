#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
// PPMͼ���ʽ�ļ򵥶�ȡ����
bool readPPM(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data) {
    std::ifstream input_file("C:/Users/office/Desktop/̬ͬ�˲�/349293-20181010164333331-2001442514.ppm"); 
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

// PPMͼ���ʽ�ļ�д�뺯��
bool writePPM(const std::string& filename, int width, int height, const std::vector<unsigned char>& data) {
    std::ofstream ofs("C:/Users/office/Desktop/̬ͬ�˲�");
    if (!ofs.is_open()) {
        return false;
    }

    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    ofs.close();

    return true;
}

// ����ɫͼ��ת��Ϊ�Ҷ�ͼ��
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

// ��ͼ���е�ÿ������Ӧ�ö����任
void applyLogTransform(std::vector<double>& input, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            input[i * width + j] = std::log(input[i * width + j] + 1.0);
        }
    }
}

// ʹ�ø���Ҷ�任����ͼ��Ķ�ά��ɢ����Ҷ�任
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

// ʹ�ø���Ҷ�任����ͼ��Ķ�ά����Ҷ��任
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
    // ��ȡ����ͼ��
    int width, height;
    std::vector<unsigned char> input_data;
    if (!readPPM("input.ppm", width, height, input_data)) {
        return 1;
	}
	std::cout<<"1";
    // ����ɫͼ��ת��Ϊ�Ҷ�ͼ��
    std::vector<double> grayscale_data;
    convertToGrayscale(input_data, width, height, grayscale_data);
	std::cout<<"2";
    // Ӧ�ö����任
    applyLogTransform(grayscale_data, width, height);
	std::cout<<"3";
    // ������ɢ����Ҷ�任
    std::vector<std::vector<std::complex<double> > > dft_data;
    computeDFT(grayscale_data, width, height, dft_data);
	std::cout<<"4";
    // Ƶ��ƽ��
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int shift_i = (i + height / 2) % height;
            int shift_j = (j + width / 2) % width;
            std::swap(dft_data[i][j], dft_data[shift_i][shift_j]);
        }
    }
	std::cout<<"5";
    // ��������ɢ����Ҷ�任
    std::vector<double> idft_data;
    computeIDFT(dft_data, width, height, idft_data);
	std::cout<<"6";
    // ����任���Ӧ��ָ���任
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            idft_data[i * width + j] = std::exp(idft_data[i * width + j]) - 1.0;
        }
    }
	std::cout<<"7";
    // �����ͼ��д���ļ�
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
