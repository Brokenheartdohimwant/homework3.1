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

bool ImageSource::openCamera(int index, int width, int height)
{
    release();

    camera_.open(index);

    if (!camera_.isOpened())
    {
        std::cerr << "错误：无法打开摄像头（编号 " << index << "）。" << std::endl;
        std::cerr << "虚拟机里要先在 VMware 菜单「虚拟机 → 可移动设备 → 摄像头 → 连接」。" << std::endl;
        std::cerr << "也可以用 --image 图片路径 的方式输入视觉标签。" << std::endl;
        return false;
    }

    // 分辨率设小一点，CPU 检测更快；驱动不接受就退回默认值
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    // 尽量关掉自动对焦抖动（不是所有摄像头都支持，失败也无所谓）
    camera_.set(cv::CAP_PROP_AUTOFOCUS, 1);

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
