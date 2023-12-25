#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <algorithm>
 
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
 
// ��RGBͼ��ת��ΪYUVͼ�� (input: image_data, output: yuv_image)
/* ע��
 * ��ɫͼ��ʹ�� R, G, B �洢����Χ���� 0~255 ֮�䣬������ unsigned char �洢�ȽϺ��ʣ�
 * ��YUVͼ���Yͨ��ֵ������Ҫ���к������Ӽ��㣬��Ҫ�þ�ȷ��С�������ʽ��ʾ������ double �洢��
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
 
// ��YUVͼ���Yͨ�����ж����任  (input: yuv_image, output: log_y_channel)
void applyLogTransform(std::vector<double>& input, int width, int height, std::vector<double>& output) {
    output.resize(width * height);
    // �� Y ͨ��Ӧ�ö����任��std::log1p(x) = log(1 + x)
    for (int i = 0; i < width * height; i++) {
        output[i] = std::log1p(input[i * 3]);
    }
}
 
// ʹ��FFTW�����Yͨ���Ķ�ά��ɢ����Ҷ�任    (input: log_y_channel, output: fft_output)
void computeFFT(const std::vector<double>& input, int width, int height, std::vector<std::complex<double> >& output) {
    // �� output ��С����Ϊͼ���С   
    output.resize(width * height);
    // input_data: �����任��ĻҶ�ͼ�� �ĸ�����ʾ������Ҷ�任ǰ��
    fftw_complex* input_data = fftw_alloc_complex(width * height);
    // output_data: ���ͼ��ĸ�����ʾ������Ҷ�任��
    fftw_complex* output_data = reinterpret_cast<fftw_complex*>(output.data());
 
    // �������任��ĻҶ�ͼ�� input ���Ƶ� input_data ���� (double���� �� ��������)������������ͼ����ʵ�������Խ��������鲿����Ϊ0��
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            input_data[index][0] = input[index];
            input_data[index][1] = 0.0;
        }
    }
 
    // ����һ��������Ҷ�任�ƻ���ִ��
    fftw_plan plan = fftw_plan_dft_2d(height, width, input_data, output_data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_free(input_data);
}
 
// ̬ͬ�˲���
double homomorphicFilter(double u, double v, double D0, double gamma_l, double gamma_h, double c) {
    double D2 = u * u + v * v;
    double filter_value = (gamma_h - gamma_l) * (1 - std::exp(-c * (D2 / (D0 * D0 * 2)))) + gamma_l;
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
            double filter_value = homomorphicFilter(u, v, D0, gamma_l, gamma_h, c);
 
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
 
// ��ԭYUVͼ���Yͨ���е�ÿ������Ӧ��������任
void applyInverseLogTransform(std::vector<double>& input, int width, int height, std::vector<double>& output) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            output[index * 3] = std::exp(input[index]) - 1.0;
        }
    }
}
 
// ��һ��Yͨ��
void normalizeYChannel(std::vector<double>& yuv_image, int width, int height) {
    // �ҵ�Yͨ�������ֵ����Сֵ
    double min_val = yuv_image[0];
    double max_val = yuv_image[0];
    for (int i = 0; i < width * height; i++) {
        double y = yuv_image[i * 3];
        min_val = std::min(min_val, y);
        max_val = std::max(max_val, y);
    }
 
    // ��Yͨ�����й�һ��
    for (int i = 0; i < width * height; i++) {
        double y = yuv_image[i * 3];
        yuv_image[i * 3] = 255.0 * (y - min_val) / (max_val - min_val);
    }
}
 
// ��YUVͼ��ת����RGBͼ�� (input: yuv_image, output: rgb_image)
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
    // ͼ����Ϣ
    int width, height;
    std::vector<unsigned char> image_data;
    std::string input_file = "input_image.ppm";
    std::string output_file = "output_image.ppm";
 
    // ��˹̬ͬ�˲�������
    double D0 = 50.0;       // ��ֹƵ��
    double gamma_l = 0.25;  // ��Ƶ����ϵ��
    double gamma_h = 1.5;   // ��Ƶ����ϵ��
    double c = 1.0;         // ��˹�˲�������״����
 
    // 1. ��ȡPPMͼ��
    if (!readPPM(input_file, width, height, image_data)) {
        std::cout << "Step  1: ��ȡPPMͼ��\t\t\t\t\tʧ��" << std::endl;
        return 1;
    }
    std::cout << "Step  1: ��ȡPPMͼ��\t\t\t\t\t�ɹ�" << std::endl;
 
    // 2. ��RGBͼ��ת��ΪYUVͼ��
    std::vector<double> yuv_image;
    convertToYUV(image_data, width, height, yuv_image);
    std::cout << "Step  2: RGBͼ��ת��ΪYUVͼ��\t\t\t\t�ɹ�" << std::endl;
 
    // 3. ��YUVͼ���Yͨ�����ж����任���洢�� log_y_channel ��
    std::vector<double> log_y_channel;
    applyLogTransform(yuv_image, width, height, log_y_channel);
    std::cout << "Step  3: ��YUVͼ���Yͨ�����ж����任\t\t\t�ɹ�" << std::endl;
 
    // 4. ʹ��FFTW�����Yͨ���Ķ�ά��ɢ����Ҷ�任
    std::vector<std::complex<double>> fft_output;
    computeFFT(log_y_channel, width, height, fft_output);
    std::cout << "Step  4: ʹ��FFTW�����Yͨ���Ķ�ά��ɢ����Ҷ�任\t�ɹ�" << std::endl;
 
    // 5. ���˲���ϵ���븵��Ҷ�任���ʵ�����鲿�������
    applyFilter(fft_output, width, height, D0, gamma_l, gamma_h, c);
    std::cout << "Step  5: ���˲���ϵ���븵��Ҷ�任���ʵ�����鲿�������\t�ɹ�" << std::endl;
 
    // 6. �����渵��Ҷ�任
    std::vector<double> ifft_output;
    computeInverseFFT(fft_output, width, height, ifft_output);
    std::cout << "Step  6: �����渵��Ҷ�任\t\t\t\t�ɹ�" << std::endl;
 
    // 7. �� ifft_output Ӧ��������任�����俽���� yuv_image ��Yͨ����
    applyInverseLogTransform(ifft_output, width, height, yuv_image);
    std::cout << "Step  7: ������任�������俽����Yͨ����\t\t�ɹ�" << std::endl;
 
    // 8. ��һ��Yͨ��
    normalizeYChannel(yuv_image, width, height);
    std::cout << "Step  8: ��һ��Yͨ��\t\t\t\t\t�ɹ�" << std::endl;
 
    // 9. ��������YUVͼ��ת����RGBͼ��
    std::vector<unsigned char> output_image;
    convertToRGB(yuv_image, width, height, output_image);
    std::cout << "Step  9: YUVͼ��ת����RGBͼ��\t\t\t\t�ɹ�" << std::endl;
 
    // 10. �������ĻҶ�ͼ�񱣴�ΪPPM��ʽ
    if (!writePPM(output_file, width, height, output_image)) {
        std::cout << "Step 10: �������ĻҶ�ͼ�񱣴�ΪPPM��ʽ\t\t\tʧ��" << std::endl;
        return 1;
    }
    std::cout << "Step 10: ��������RGBͼ�񱣴�ΪPPM��ʽ\t\t\t�ɹ�" << std::endl;
 
    return 0;
}