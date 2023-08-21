#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>

// PPMͼ���ʽ�ļ򵥶�ȡ����
bool readPPM(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data) {
    // �Զ�����ģʽ���ļ���ͼ��Ϊһ���������ļ���
    std::ifstream input_file(filename, std::ios::binary);
    // ����ܷ�򿪸��ļ�
    if (!input_file.is_open()) {
        std::cout << "�޷����ļ�:  " << filename << std::endl;
        return false;
    }

    // ���ħ����PPM��ħ����"P6"���������P6����PPM��ʽ���ļ���
    std::string magic_number;
    input_file >> magic_number;
    if (magic_number != "P6") {
        std::cout << "�ļ����� PPM ��ʽ: " << filename << std::endl;
        return false;
    }

    // ��ȡͼ��Ŀ�ߣ��Լ�ͼ��������ɫֵ max_value
    input_file >> width >> height;
    int max_value;
    input_file >> max_value;
    input_file.ignore(1);

    // ���� image_data �Ĵ�С����Ӧͼ���С��ÿ�������� 3 ����ɫ��������ɫ����ɫ����ɫ����ɣ�����ÿ������ռ�� 3 ���ֽڣ�
    data.resize(width * height * 3);

    /* reinterpret_cast<char*>(data.data()) :
       ����һ������ת������ data.data() ���ص� unsigned char ָ��ת��Ϊ char ָ�롣
       ��Ϊ read() ������Ҫһ�� char ���͵�ָ����Ϊ������������Ҫ��������ת����

       ���д���� PPM �ļ��ж�ȡ width * height * 3 ���ֽڵ��������ݣ�
       ������Щ���ݴ洢�� data �����С�������data �����Ͱ�����ͼ��������������ݡ�
    */
    input_file.read(reinterpret_cast<char*>(data.data()), data.size());

    return true;
}

// PPMͼ���ʽ�ļ�д�뺯��
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

// ����ɫͼ��ת��Ϊ�Ҷ�ͼ�� (input: image_data, output: grayscale_image)
/* ע��
 * ��ɫͼ��ʹ�� R, G, B �洢����Χ���� 0~255 ֮�䣬������ unsigned char �洢�ȽϺ��ʣ�
 * ���Ҷ�ֵ������Ҫ���к������Ӽ��㣬��Ҫ�þ�ȷ��С�������ʽ��ʾ������ double �洢��
 */
void convertToGrayscale(const std::vector<unsigned char>& input, int width, int height, std::vector<double>& output) {
    // �Ҷ�ͼ��ÿ������ռ 1 ���ֽ�
    output.resize(width * height);
    for (int i = 0; i < width * height; i++) {
        // ����ÿ�����صĻҶ�ֵ
        float r = input[i * 3];
        float g = input[i * 3 + 1];
        float b = input[i * 3 + 2];
        // ��������ĻҶ�ֵ�洢�� grayscale_image ������
        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;  // luma ת����ʽ
    }
}

// ��ͼ���е�ÿ������Ӧ�ö����任  (input: grayscale_image)
void applyLogTransform(std::vector<double>& input, int width, int height) {
    // ��ÿ������Ӧ�ö����任��std::log1p(x) = log(1 + x)
    for (int i = 0; i < width * height; i++) {
        input[i] = std::log1p(input[i]);
    }
}

// ʹ��FFTW�����ͼ��Ķ�ά��ɢ����Ҷ�任    (input: grayscale_image, output: fft_output)
void computeFFT(const std::vector<double>& input, int width, int height, std::vector<std::complex<double> >& output) {
    // �� output ��С����Ϊͼ���С   
    output.resize(width * height);
    // input_data: �����任��ĻҶ�ͼ�� �ĸ�����ʾ������Ҷ�任ǰ��
    fftw_complex* input_data = fftw_alloc_complex(width * height);
    // output_data: ���ͼ��ĸ�����ʾ������Ҷ�任��
    fftw_complex* output_data = reinterpret_cast<fftw_complex*>(output.data());
    // ����һ��������Ҷ�任�ƻ�
    fftw_plan plan = fftw_plan_dft_2d(height, width, input_data, output_data, FFTW_FORWARD, FFTW_ESTIMATE);

    // �������任��ĻҶ�ͼ�� input ���Ƶ� input_data ���� (double���� �� ��������)������������ͼ����ʵ�������Խ��������鲿����Ϊ0��
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input_data[index][0] = input[index];
            input_data[index][1] = 0.0;
        }
    }

    fftw_execute(plan); // ִ�мƻ������б任
    fftw_destroy_plan(plan);
    fftw_free(input_data);
}

// ��˹̬ͬ�˲���
double gaussianHomomorphicFilter(double u, double v, double D0, double gamma_l, double gamma_h, double c) {
    double D2 = u * u + v * v;
    double filter_value = (gamma_h - gamma_l) * (1 - std::exp(-c * (D2 / (D0 * D0)))) + gamma_l;
    return filter_value;
}

// ���˲���ϵ���븵��Ҷ�任���ʵ�����鲿�������
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

// �����渵��Ҷ�任
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

// ��ͼ���е�ÿ������Ӧ��������任
void applyInverseLogTransform(std::vector<double>& input, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input[index] = std::exp(input[index]) - 1.0;
        }
    }
}

// ���Ҷ�ͼ������ӳ�䵽0-255��Χ
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
    std::string input_file = "F:\\����\\HomomorphicFiltering\\x64\\Debug\\input_image.ppm";
    std::string output_file = "F:\\����\\HomomorphicFiltering\\x64\\Debug\\output_image.ppm";

    // 0. ��ȡPPMͼ��
    if (!readPPM(input_file, width, height, image_data)) {
        std::cout << "Step 0: ��ȡPPMͼ��\t\t\t\t\tʧ��" << std::endl;
        return 1;
    }
    std::cout << "Step 0: ��ȡPPMͼ��\t\t\t\t\t�ɹ�" << std::endl;

    // 1. ����ɫͼ��ת��Ϊ�Ҷ�ͼ��
    std::vector<double> grayscale_image;
    convertToGrayscale(image_data, width, height, grayscale_image);
    std::cout << "Step 1: ����ɫͼ��ת��Ϊ�Ҷ�ͼ��\t\t\t�ɹ�" << std::endl;

    // 2. ��ͼ���е�ÿ������Ӧ�ö����任
    applyLogTransform(grayscale_image, width, height);
    std::cout << "Step 2: ��ͼ���е�ÿ������Ӧ�ö����任\t\t\t�ɹ�" << std::endl;

    // 3. ʹ��FFTW�����ͼ��Ķ�ά��ɢ����Ҷ�任
    std::vector<std::complex<double> > fft_output;
    computeFFT(grayscale_image, width, height, fft_output);
    std::cout << "Step 3: ʹ��FFTW�����ͼ��Ķ�ά��ɢ����Ҷ�任\t\t�ɹ�" << std::endl;

    // ��˹̬ͬ�˲�������
    double D0 = 50.0;
    double gamma_l = 0.25;
    double gamma_h = 1.5;
    double c = 1.0;

    // 4. ���˲���ϵ���븵��Ҷ�任���ʵ�����鲿�������
    applyFilter(fft_output, width, height, D0, gamma_l, gamma_h, c);
    std::cout << "Step 4: ���˲���ϵ���븵��Ҷ�任���ʵ�����鲿�������\t�ɹ�" << std::endl;

    // 5. �����渵��Ҷ�任
    std::vector<double> processed_image;
    computeInverseFFT(fft_output, width, height, processed_image);
    std::cout << "Step 5: �����渵��Ҷ�任\t\t\t\t�ɹ�" << std::endl;

    // 6. ��ͼ���е�ÿ������Ӧ��������任
    applyInverseLogTransform(processed_image, width, height);
    std::cout << "Step 6: ��ͼ���е�ÿ������Ӧ��������任\t\t�ɹ�" << std::endl;

    // 7. ���Ҷ�ͼ������ӳ�䵽0-255��Χ
    std::vector<unsigned char> output_image;
    remapGrayscaleImage(processed_image, width, height, output_image);
    std::cout << "Step 7: ���Ҷ�ͼ������ӳ�䵽0-255��Χ\t\t\t�ɹ�" << std::endl;

    // 8. �������ĻҶ�ͼ�񱣴�ΪPPM��ʽ
    if (!writePPM(output_file, width, height, output_image)) {
        std::cout << "Step 8: �������ĻҶ�ͼ�񱣴�ΪPPM��ʽ\t\t\tʧ��" << std::endl;
        return 1;
    }
    std::cout << "Step 8: �������ĻҶ�ͼ�񱣴�ΪPPM��ʽ\t\t\t�ɹ�" << std::endl;

    return 0;
}
