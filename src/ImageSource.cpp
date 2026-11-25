#include "ImageSource.hpp"

#include <iostream>

bool ImageSource::loadImage(const std::string& path, cv::Mat& image)
{
    image = cv::imread(path, cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr << "无法读取图片：" << path << std::endl;
        return false;
    }

    return true;
}

bool ImageSource::openCamera(int index)
{
    release();

    camera_.open(index);

    if (!camera_.isOpened())
    {
        std::cerr << "错误：无法打开摄像头（编号 " << index << "）。" << std::endl;
        std::cerr << "可以用 --image 图片路径 的方式输入视觉标签。" << std::endl;
        return false;
    }

    camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    return true;
}

bool ImageSource::read(cv::Mat& frame)
{
    if (!camera_.isOpened())
    {
        return false;
    }

    if (!camera_.read(frame) || frame.empty())
    {
        return false;
    }

    return true;
}

bool ImageSource::isCameraOpen() const
{
    return camera_.isOpened();
}

void ImageSource::release()
{
    if (camera_.isOpened())
    {
        camera_.release();
    }
}