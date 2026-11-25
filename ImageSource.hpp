#ifndef IMAGE_SOURCE_HPP
#define IMAGE_SOURCE_HPP

#include <opencv2/opencv.hpp>

#include <string>

// 图像来源：图片文件 或 摄像头
class ImageSource
{
public:
    bool loadImage(const std::string& path, cv::Mat& image);

    bool openCamera(int index = 0, int width = 1280, int height = 720);
    bool read(cv::Mat& frame);
    bool isCameraOpen() const;
    void release();

private:
    cv::VideoCapture camera_;
};

#endif
