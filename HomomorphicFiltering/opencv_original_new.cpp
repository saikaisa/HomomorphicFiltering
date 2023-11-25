#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;

// �������
void input(double& gammaH, double& gammaL, double& C, double& d0) {
    while (1) {
        cout << "���� -1 ʹ��Ĭ�ϲ��������� 0 ��д������";
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

// ��ʾͼ��
void showImage(const string& windowName, const Mat& image, int displayWidth, int windowPosX, int windowPosY) {
    // ����һ�����ڣ������ô��ڱ�־
    namedWindow(windowName, WINDOW_NORMAL);

    // ���ô��ڵ�λ��
    moveWindow(windowName, windowPosX, windowPosY);

    // ����������
    float scale = (float)displayWidth / image.cols;
    resizeWindow(windowName, displayWidth, image.rows * scale);

    imshow(windowName, image);
}

// ����Ҷ�任�õ�Ƶ��ͼ�͸�������
void custom_dft(Mat input_image, Mat& spectrogram, Mat& fft_image) {
    // 1.��չͼ�����Ϊ2��3��5�ı���ʱ�����ٶȿ�
    int m = getOptimalDFTSize(input_image.rows);
    int n = getOptimalDFTSize(input_image.cols);
    copyMakeBorder(input_image, input_image, 0, m - input_image.rows, 0, n - input_image.cols, BORDER_CONSTANT, Scalar::all(0));

    // 2.����һ��˫ͨ������planes���������渴����ʵ�����鲿
    Mat planes[] = { Mat_<double>(input_image), Mat::zeros(input_image.size(), CV_64F) };

    // 3.�Ӷ����ͨ�������д���һ����ͨ������:fft_image������Merge����������ϲ�Ϊһ����ͨ�����У�����������ÿ��Ԫ�ؽ�����������Ԫ�صļ���
    merge(planes, 2, fft_image);

    // 4.���и���Ҷ�任
    dft(fft_image, fft_image);

    // 5.���㸴���ķ�ֵ��������spectrogram��Ƶ��ͼ��
    split(fft_image, planes);   // ��˫ͨ����Ϊ������ͨ����һ����ʾʵ����һ����ʾ�鲿
    Mat fft_image_real = planes[0];
    Mat fft_image_imag = planes[1];

    magnitude(planes[0], planes[1], spectrogram);   // ���㸴���ķ�ֵ��������spectrogram��Ƶ��ͼ��

    // 6.ǰ��õ���Ƶ��ͼ�������󣬲�����ʾ�����ת��
    spectrogram += Scalar(1);   // ȡ����ǰ�����е����ض��� 1����ֹlog0
    log(spectrogram, spectrogram);  // ȡ����
    normalize(spectrogram, spectrogram, 0, 1, NORM_MINMAX); // ��һ��

    // 7.���к��طֲ�����ͼ����
    spectrogram = spectrogram(Rect(0, 0, spectrogram.cols & -2, spectrogram.rows & -2));

    // ��ʾ���Ļ�֮ǰ��Ƶ��ͼ
    showImage("���Ļ�ǰ��Ƶ��ͼ", spectrogram, 400, 100, 450);

    // �������и���Ҷͼ���е����ޣ�ʹԭ��λ��ͼ������
    int cx = spectrogram.cols / 2;
    int cy = spectrogram.rows / 2;
    Mat q0(spectrogram, Rect(0, 0, cx, cy));   // ��������
    Mat q1(spectrogram, Rect(cx, 0, cx, cy));  // ��������
    Mat q2(spectrogram, Rect(0, cy, cx, cy));  // ��������
    Mat q3(spectrogram, Rect(cx, cy, cx, cy)); // ��������

    // �����������Ļ�
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);  // ���������½��н���
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);  // ���������½��н���

    Mat q00(fft_image_real, Rect(0, 0, cx, cy));    // ��������
    Mat q01(fft_image_real, Rect(cx, 0, cx, cy));   // ��������
    Mat q02(fft_image_real, Rect(0, cy, cx, cy));   // ��������
    Mat q03(fft_image_real, Rect(cx, cy, cx, cy));  // ��������
    q00.copyTo(tmp); q03.copyTo(q00); tmp.copyTo(q03);  // ���������½��н���
    q01.copyTo(tmp); q02.copyTo(q01); tmp.copyTo(q02);  // ���������½��н���

    Mat q10(fft_image_imag, Rect(0, 0, cx, cy));    // ��������
    Mat q11(fft_image_imag, Rect(cx, 0, cx, cy));   // ��������
    Mat q12(fft_image_imag, Rect(0, cy, cx, cy));   // ��������
    Mat q13(fft_image_imag, Rect(cx, cy, cx, cy));  // ��������
    q10.copyTo(tmp); q13.copyTo(q10); tmp.copyTo(q13);  // ���������½��н���
    q11.copyTo(tmp); q12.copyTo(q11); tmp.copyTo(q12);  // ���������½��н���

    planes[0] = fft_image_real;
    planes[1] = fft_image_imag;
    merge(planes, 2, fft_image);    	//������Ҷ�任������Ļ�
}

int main() {
    // ��ȡԭʼͼ��
    Mat image_raw = imread("color_image.jpg");
    if (image_raw.empty()) {
        cout << "��ȡ����" << endl;
        return -1;
    }

    // �������
    double gammaH = 2.0;
    double gammaL = 0.2;
    double C = 0.1;       // Խ��Խ��
    double d0 = max(image_raw.cols, image_raw.rows);    // �˲��뾶
    input(gammaH, gammaL, C, d0);
    // ���ú�����ʾͼ��
    showImage("ԭʼͼ��", image_raw, 400, 100, 100);

    // ת��ΪYUV��ɫ�ռ�
    cvtColor(image_raw, image_raw, COLOR_BGR2YUV);  // ԭRGB����YUV
    vector<Mat> image_yuv;    // ���ڴ洢 3 ����ͨ��ͼ�񣨷ֱ�ΪY��U��Vͨ����
    split(image_raw, image_yuv);    // ��һ����ͨ��ͼ��image_raw ����Ϊ3����ͨ��ͼ��image_yuv[0],image_yuv[1],image_yuv[2]
    Mat image_y = image_yuv[0];  // image_y�洢���� Y ͨ��ͼ��ֻʣ��������Ϣ��ÿ������ֵ��ֻ��ʾ�ڰ׻Ҷ�

    // ���ж����任
    Mat image_ln(image_y.size(), CV_64F);
    for (int i = 0; i < image_y.rows; i++) {
        for (int j = 0; j < image_y.cols; j++) {
            image_ln.at<double>(i, j) = log(image_y.at<uchar>(i, j) + 0.1);
        }
    }

    // ����Ҷ�任��image_spectrogram Ϊ����ʾ��Ƶ��ͼ��image_fft Ϊ����Ҷ�任�ĸ������
    Mat image_spectrogram, image_fft;
    custom_dft(image_ln, image_spectrogram, image_fft);
    showImage("���Ļ����Ƶ��ͼ", image_spectrogram, 400, 510, 450);

    // ���и�˹̬ͬ�˲�
    Mat planes[] = { Mat_<double>(image_spectrogram), Mat::zeros(image_spectrogram.size(),CV_64F) };
    split(image_fft, planes);   // ����ͨ������ȡʵ���鲿
    Mat image_fft_real = planes[0];
    Mat image_fft_imag = planes[1];

    // Ƶ��ͼ��������
    int center_x = image_fft_real.cols / 2;
    int center_y = image_fft_real.rows / 2;

    // ������˹̬ͬ�˲�����
    Mat h = Mat::zeros(image_fft_real.size(), CV_64F);

    // x Ϊ�У�y Ϊ��
    for (int y = 0; y < image_fft_real.rows; y++) {
        double* y_pt = h.ptr<double>(y);   // ָ���е�ָ��
        for (int x = 0; x < image_fft_real.cols; x++) {
            int distance_sq = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
            y_pt[x] = (gammaH - gammaL) * (1 - exp(-C * distance_sq / (d0 * d0))) + gammaL;
        }
    }

    // �����Ӧ�������
    image_fft_real = image_fft_real.mul(h);
    image_fft_imag = image_fft_imag.mul(h);
    planes[0] = image_fft_real;
    planes[1] = image_fft_imag;

    // ��˹̬ͬ�˲����
    Mat image_homo_res;
    merge(planes, 2, image_homo_res);

    // ����Ҷ��任
    Mat iDft[] = { Mat_<double>(image_spectrogram), Mat::zeros(image_spectrogram.size(), CV_64F) };
    idft(image_homo_res, image_homo_res);   // ����Ҷ��任���õ�ʵ�����鲿
    split(image_homo_res, iDft);    // ����ͨ������Ҫ��ȡ 0 ͨ������ʵ��

    // ���㸴���ķ�ֵ����ֵ = sqrt(ʵ��^2 + �鲿^2)���������� iDft[0]
    // �õ�ͼ�������ǿ�ȣ�ͨ����ֵ���Ի�ԭԭʼͼ���������Ϣ
    magnitude(iDft[0], iDft[1], iDft[0]);

    // ��һ������������ֵ����ָ���任�����(inf)
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);

    // ָ���任
    exp(iDft[0], iDft[0]);

    // ��һ������ʹͼ�������� 0~255 ��Χ��
    normalize(iDft[0], iDft[0], 0, 255, NORM_MINMAX, CV_8U);

    // �ϲ�ͨ����ת����BGR��ɫ�ռ�
    Mat result(image_raw.rows, image_raw.cols, CV_8UC3);
    vector<Mat> yuv_channels = { iDft[0], image_yuv[1], image_yuv[2] };
    merge(yuv_channels, result);    // ��split�෴��merge�ǽ�3����ͨ��ͼ����ϳ�1����ͨ��ͼ��
    cvtColor(result, result, COLOR_YUV2BGR);  // ԭ YUV���� RGB

    showImage("���", result, 400, 510, 100); // ��ʾ�������ͼ��
    waitKey(0);

    return 0;
}