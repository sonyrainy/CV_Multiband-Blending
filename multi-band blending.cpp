
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world4100d")
#else
#pragma comment(lib, "opencv_world4100")
#endif

using namespace cv;
using namespace std;

vector<Mat> buildLaplacian(const Mat& img, int depth) {
    vector<Mat> pyr(depth);
    pyr[0] = img.clone();

    for (int i = 0; i < depth - 1; i++) {
        //다시 업샘플링 하면서 resolution은 올라가지만, 가지고 있는 주파수는 변하지 않기 때문에
        //다운샘플링 되며 사라졌던 high 주파수는 다시 생성되지 않는다.
        //+) i가 낮아질수록 (이미지 크기가 같다면) high가 더 뚜렷하게 보이게 된다.
        Mat tmp;
        pyrDown(pyr[i], pyr[i + 1]);
        pyrUp(pyr[i + 1], tmp, pyr[i].size());
        pyr[i] = pyr[i] - tmp;
    }

    return pyr;
}

vector<Mat> buildGaussian(const Mat& img, int depth) {
    vector<Mat> pyr(depth);
    pyr[0] = img.clone();

    for (int i = 0; i < depth - 1; i++) {
        pyrDown(pyr[i], pyr[i + 1]);
    }
    return pyr;
}

Mat reconstruct(const vector<Mat>& pyr) {
    Mat img = pyr.back().clone();
    for (int i = 0; i <= pyr.size() - 2; i++) {
        Mat tmp;
        pyrUp(img, tmp, pyr[pyr.size() - 2 - i].size());
        img = tmp + pyr[pyr.size() - 2 - i];
    }
    return img;
}


int main() {
    int depth = 8;

    Mat apple = imread("C:/Users/sonyrainy/Desktop/burt_apple.png");
    Mat orange = imread("C:/Users/sonyrainy/Desktop/burt_orange.png");
    Mat mask = imread("C:/Users/sonyrainy/Desktop/burt_mask.png");


    apple.convertTo(apple, CV_32F, 1.0 / 255.0);
    orange.convertTo(orange, CV_32F, 1.0 / 255.0);
    mask.convertTo(mask, CV_32F, 1.0 / 255.0);

    Mat mask2 = Scalar(1, 1, 1) - mask;

    vector<Mat> appleLapPyr = buildLaplacian(apple, depth);
    vector<Mat> orangeLapPyr = buildLaplacian(orange, depth);
    vector<Mat> maskGausPyr = buildGaussian(mask, depth);
    vector<Mat> mask2GausPyr = buildGaussian(mask2, depth);

    vector<Mat> added(depth);

    for (int i = 0; i < depth; i++) {
        Mat apple_Masked, orange_Masked;
        Mat upSampleMasked, upSampleMasked2;


        //multiply 연산 시, 연산 가능하도록 행렬 형태 맞추기 위해서 추가
        if (maskGausPyr[i].size() != appleLapPyr[i].size()) {
            pyrUp(maskGausPyr[i], upSampleMasked, appleLapPyr[i].size());
        }
        else {
            upSampleMasked = maskGausPyr[i];
        }


        if (mask2GausPyr[i].size() != orangeLapPyr[i].size()) {
            pyrUp(mask2GausPyr[i], upSampleMasked2, orangeLapPyr[i].size());
        }
        else {
            upSampleMasked2 = mask2GausPyr[i];
        }

        multiply(appleLapPyr[i], upSampleMasked, apple_Masked);
        multiply(orangeLapPyr[i], upSampleMasked2, orange_Masked);
        added[i] = apple_Masked + orange_Masked;
    }

    Mat result = reconstruct(added);

    imshow("result", result);
    waitKey(0);

    return 0;
}
